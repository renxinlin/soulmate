# -*- coding: utf-8 -*-
"""
调度器抽象层
============
整个调度框架的基石，定义：

* ``TaskState``    —— 任务状态机
* ``TaskContext``  —— 传递给任务 Callback 与钩子函数的上下文（携带 stop_event 与信号量）
* ``BaseTask``     —— 所有任务类型的抽象基类（统一超时、钩子、执行、持久化骨架）

设计要点
--------
1. 超时机制：每个任务持有独立的 ``stop_event``。调度器在任务开始执行时启动一个
   ``threading.Timer``，超时后置位 ``stop_event``；回调函数内可通过
   ``ctx.stop_event.is_set()`` 协作式退出。线程无法被强行杀死，因此超时是
   "协作式 + 状态标记" 的，未协作的回调会继续占用一个工作线程直到自行返回。
2. 信号量：任务真正执行回调前会先获取调度器的全局信号量（限制并发），并把该
   信号量连同 stop_event 一起通过 ``TaskContext`` 传入回调，回调可感知并发水位。
3. 钩子回调：start / end / timeout / error / cancel，既可子类覆写，也可在
   构造时通过 ``hooks={"start": fn, ...}`` 注入（注入优先级更高）。
4. 持久化：``to_dict()`` 只保存 JSON 结构元信息；回调/钩子以 "模块.限定名"
   字符串保存，重启后通过 ``from_dict()`` 重新 import 还原。
"""
from __future__ import annotations

import enum
import importlib
import inspect
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TaskState(enum.Enum):
    """任务状态机"""

    PENDING = "pending"      # 已提交，等待触发
    RUNNING = "running"      # 正在执行回调
    SUCCESS = "success"      # 执行成功
    FAILED = "failed"        # 执行失败（回调抛异常）
    TIMEOUT = "timeout"      # 超时（stop_event 被置位）
    CANCELLED = "cancelled"  # 已取消

    @property
    def terminal(self) -> bool:
        """是否处于终态（不再参与调度）"""
        return self in (TaskState.SUCCESS, TaskState.FAILED,
                        TaskState.TIMEOUT, TaskState.CANCELLED)


class TaskContext:
    """
    任务执行上下文，是 Callback / 钩子函数的统一入参。

    Attributes
    ----------
    task       : 当前任务对象
    stop_event : 超时/取消信号量事件（回调应周期性检查并退出）
    semaphore  : 调度器全局并发信号量（已随任务进入）
    result     : 回调返回值（成功后写入）
    error      : 回调异常（失败后写入）
    start_time / end_time : 起止时间戳
    """

    __slots__ = ("task", "stop_event", "semaphore",
                 "result", "error", "start_time", "end_time")

    def __init__(self, task: "BaseTask", stop_event: threading.Event,
                 semaphore: threading.Semaphore) -> None:
        self.task = task
        self.stop_event = stop_event
        self.semaphore = semaphore
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def timed_out(self) -> bool:
        return self.stop_event.is_set()

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"<TaskContext task={self.task.name} "
                f"stop_event={self.stop_event.is_set()}>")


def to_epoch(value: Any) -> Optional[float]:
    """把 datetime / 时间戳 / 字符串统一转成 epoch 秒。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).timestamp()
            except ValueError:
                continue
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"无法解析的时间字符串: {value!r}")
    raise TypeError(f"不支持的时间类型: {type(value).__name__}")


def qualname(obj: Optional[Callable]) -> Optional[str]:
    """取可 import 对象的 "模块.限定名"，用于 JSON 元信息持久化。"""
    if obj is None:
        return None
    mod = getattr(obj, "__module__", None)
    qn = getattr(obj, "__qualname__", None)
    if mod and qn and not mod.startswith("<"):
        return f"{mod}.{qn}"
    return None


def restore_callable(ref: Optional[str]) -> Optional[Callable]:
    """按 "模块.限定名" 重新 import 还原回调/钩子。"""
    if not ref:
        return None
    module_name, _, attr_path = ref.rpartition(".")
    if not module_name:
        raise ImportError(f"无效的可调用引用: {ref!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{ref!r} 不是可调用对象")
    return obj


class BaseTask(ABC):
    """
    任务抽象基类。

    子类需要实现：
    * ``is_periodic``   —— 是否周期任务（决定执行完成后是否重新入队）
    * ``next_run_at()`` —— 下一次触发时刻（epoch 秒）；周期任务返回下一次时间，
                           一次性任务在触发过之后返回 None
    * ``to_dict() / from_dict()`` —— 继承基类实现，补充各自字段
    """

    #: 是否为周期任务（周期任务执行完成后调度器会重新计算下一次触发时间）
    is_periodic: bool = False

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        timeout: Optional[float] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.task_id: str = task_id or uuid.uuid4().hex
        self.name: str = name
        self.callback: Optional[Callable] = callback
        self.timeout: Optional[float] = timeout        # 超时秒数；None 表示不超时
        self.hooks: Dict[str, Callable] = hooks or {}
        self.state: TaskState = TaskState.PENDING
        self.stop_event = threading.Event()
        self.created_at: float = time.time()
        self.run_at: Optional[float] = None            # 本次触发时刻（epoch）
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self._fired: bool = False
        # 回调是否接受 ctx 参数（兼容无参回调）
        self._accepts_ctx = self._probe_accepts_ctx(self.callback)

    # ------------------------------------------------------------------
    # 钩子回调
    # ------------------------------------------------------------------
    def on_start(self, ctx: TaskContext) -> None:
        self._fire_hook("start", ctx)

    def on_end(self, ctx: TaskContext) -> None:
        self._fire_hook("end", ctx)

    def on_timeout(self, ctx: TaskContext) -> None:
        self._fire_hook("timeout", ctx)

    def on_error(self, ctx: TaskContext) -> None:
        self._fire_hook("error", ctx)

    def on_cancel(self, ctx: TaskContext) -> None:
        self._fire_hook("cancel", ctx)

    def _fire_hook(self, name: str, ctx: TaskContext) -> None:
        hook = self.hooks.get(name)
        if hook is None:
            return
        try:
            hook(ctx)
        except Exception:
            logger.exception("任务 %s 的钩子 %s 执行失败", self.task_id, name)

    # ------------------------------------------------------------------
    # 超时 / 取消 / 复位
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        """取消任务：置位 stop_event（运行中的回调可协作退出）。"""
        self.stop_event.set()
        self.state = TaskState.CANCELLED

    def reset(self) -> None:
        """周期任务复用前复位（重新生成 stop_event）。"""
        self.stop_event = threading.Event()
        self.state = TaskState.PENDING
        self.result = None
        self.error = None
        self._fired = False

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    @staticmethod
    def _probe_accepts_ctx(fn: Optional[Callable]) -> bool:
        if fn is None:
            return False
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return True
        params = [p for p in sig.parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        return bool(params)

    def _invoke(self, ctx: TaskContext) -> Any:
        """在工作线程中真正执行用户回调；回调统一接收 TaskContext。"""
        if self.callback is None:
            return None
        if self._accepts_ctx:
            return self.callback(ctx)
        return self.callback()

    def execute(self, executor, semaphore: threading.Semaphore) -> TaskContext:
        """
        执行一次任务（由调度器的 runner 线程调用，回调提交到独立回调线程池）。

        流程：状态置 RUNNING → on_start → 获取信号量 → 提交回调到回调池 →
        轮询等待：回调完成 / 超时置位 stop_event / 被取消 →
        判定 SUCCESS / TIMEOUT / CANCELLED → 释放信号量 → on_end。

        超时语义：到达 deadline 时置位 ``stop_event``（协作信号），回调可通过
        ``ctx.stop_event`` 感知并退出；即便回调不协作，runner 也会在超时后
        立即返回并标记 TIMEOUT，残留的回调线程由线程池守护线程自行收尾。
        """
        ctx = TaskContext(self, self.stop_event, semaphore)
        self.state = TaskState.RUNNING
        ctx.start_time = time.time()
        self.on_start(ctx)

        timer = None
        acquired = False
        try:
            semaphore.acquire()
            acquired = True
            future = executor.submit(self._invoke, ctx)
            deadline = time.time() + self.timeout if self.timeout else None
            while not future.done():
                if deadline is not None and time.time() >= deadline:
                    self.stop_event.set()          # 超时：置位 stop_event
                    break
                if self.stop_event.wait(0.05):     # 被取消或超时打断
                    break
            if self.state is TaskState.CANCELLED:
                self.on_cancel(ctx)                # 外部取消（运行中）
                return ctx
            if self.stop_event.is_set() and not future.done():
                self.state = TaskState.TIMEOUT     # 回调未协作，先释放控制权
                self.on_timeout(ctx)
                return ctx
            result = future.result()
            ctx.result = result
            if self.stop_event.is_set():
                self.state = TaskState.TIMEOUT
                self.on_timeout(ctx)
            else:
                self.state = TaskState.SUCCESS
        except Exception as exc:              # noqa: BLE001 - 捕获用户回调的一切异常
            ctx.error = exc
            self.state = TaskState.FAILED
            self.on_error(ctx)
            logger.exception("任务 %s(%s) 执行失败", self.task_id, self.name)
        finally:
            ctx.end_time = time.time()
            if acquired:
                semaphore.release()
            if not self.is_periodic:
                self._fired = True
            self.on_end(ctx)
        return ctx

    # ------------------------------------------------------------------
    # 调度接口（子类实现）
    # ------------------------------------------------------------------
    @abstractmethod
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        """下一次触发时刻；返回 None 表示不再调度。"""

    # ------------------------------------------------------------------
    # 持久化（JSON 结构元信息）
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """导出 JSON 元信息（仅结构，不导出运行期对象）。"""
        return {
            "task_id": self.task_id,
            "type": type(self).__name__,
            "name": self.name,
            "timeout": self.timeout,
            "state": self.state.value,
            "created_at": self.created_at,
            "run_at": self.run_at,
            "callback": qualname(self.callback),
            "args": [],
            "kwargs": {},
            "hooks": {k: qualname(v) for k, v in self.hooks.items() if v is not None},
        }

    def _restore_state(self, data: Dict[str, Any]) -> None:
        self.created_at = data.get("created_at", time.time())
        self.run_at = data.get("run_at")
        if data.get("state"):
            self.state = TaskState(data["state"])

    @staticmethod
    def _restore_callable(ref: Optional[str]) -> Optional[Callable]:
        """按 "模块.限定名" 重新 import 还原回调/钩子。"""
        return restore_callable(ref)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseTask":
        """由 JSON 元信息重建任务（子类需补各自字段）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name} run_at={self.run_at} state={self.state.value}>"


def parse_timeout(value: Any) -> Optional[float]:
    """超时参数归一化：支持秒数或 timedelta。"""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)
