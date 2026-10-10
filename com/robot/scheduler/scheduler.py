# -*- coding: utf-8 -*-
"""
调度器 Scheduler
================
调度引擎，统一编排：

* **线程池**（``ThreadPoolExecutor``）—— 执行任务的回调
* **信号量**（``threading.Semaphore``）—— 限制并发任务数，随 TaskContext 传入回调
* **四层时间轮**（天/时/分/秒）—— 处理 7 天以内的任务
* **磁盘缓存 + 扫描器** —— 处理 > 7 天的任务（扫描任务提前一天定时检测，
  任务进入 7 天窗口后加载回内存时间轮）
* **JSON 元信息持久化** —— 内存任务落盘，重启后 ``load_state()`` 还原再调度

使用示例::

    sched = Scheduler(workers=8, max_concurrency=4)
    sched.start()
    sched.submit(ScheduledTask("job", cb, run_at="2026-10-10 12:00:00"))
    ...
    sched.save_state()          # 落盘
    sched.shutdown()

    # 重启后
    sched2 = Scheduler(...)
    sched2.start()
    sched2.load_state()         # 还原内存任务并重新入队
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from .base import BaseTask, TaskState, TaskContext
from .delay_task import DelayTask
from .delay_periodic_task import DelayPeriodicTask
from .disk_store import DiskTaskStore
from .periodic_task import PeriodicTask
from .scheduled_periodic_task import ScheduledPeriodicTask
from .scheduled_task import ScheduledTask
from .task_store import TaskStore, restore_task
from .time_wheel import HierarchicalTimeWheel

logger = logging.getLogger(__name__)

#: 默认 7 天窗口（秒）
DEFAULT_HORIZON_DAYS = 7


class Scheduler:
    """
    任务调度器。

    Parameters
    ----------
    workers         : 回调线程池大小（真正执行回调的线程数）
    runners         : runner 线程池大小（等待回调完成/超时的调度线程，默认 = workers）
    max_concurrency : 同时执行的回调数上限（信号量，默认 = workers）
    tick_interval   : 时间轮 tick 周期（秒）
    scan_interval   : 磁盘扫描周期（秒），默认 86400（每天一次，"提前一天检测"）
    horizon_days    : 磁盘缓存窗口天数（默认 7）
    disk_dir        : 远期任务磁盘缓存目录
    state_file      : 内存任务元信息 JSON 文件
    """

    def __init__(
        self,
        workers: int = 8,
        runners: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        tick_interval: float = 1.0,
        scan_interval: float = 86400.0,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        disk_dir: str = "scheduler_runtime/disk",
        state_file: str = "scheduler_runtime/state.json",
    ) -> None:
        self.workers = workers
        self.runners = runners or workers
        self.max_concurrency = max_concurrency or workers
        # 双池设计：runner 池负责"提交回调 + 等待结果/超时"，
        # 回调池真正执行用户回调，避免所有工作线程互相等待造成死锁。
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sched-callback")
        self.runner_pool = ThreadPoolExecutor(
            max_workers=self.runners, thread_name_prefix="sched-runner")
        self.semaphore = threading.Semaphore(self.max_concurrency)
        self.horizon = horizon_days * 86400
        self.scan_interval = scan_interval

        self.wheel = HierarchicalTimeWheel(
            on_due=self._on_due, tick_interval=tick_interval)
        self.disk = DiskTaskStore(disk_dir)
        self.store = TaskStore(state_file)

        self.tasks: Dict[str, BaseTask] = {}     # task_id -> 任务（含已结束记录）
        self._lock = threading.RLock()
        self._running = False
        self._scanner: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动时间轮 ticker 与磁盘扫描线程，并立即执行一次扫描。"""
        if self._running:
            return
        self._running = True
        self.wheel.start()
        self._scanner = threading.Thread(
            target=self._scan_loop, name="disk-task-scanner", daemon=True)
        self._scanner.start()
        try:
            self._scan_once()
        except Exception:
            logger.exception("启动扫描失败")

    def shutdown(self, wait: bool = True) -> None:
        """停止扫描器、时间轮并关闭线程池。"""
        self._running = False
        if self._scanner is not None:
            self._scanner.join(timeout=2)
            self._scanner = None
        self.wheel.stop()
        self.executor.shutdown(wait=wait, cancel_futures=True)
        self.runner_pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "Scheduler":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # 提交 / 取消
    # ------------------------------------------------------------------
    def submit(self, task: BaseTask) -> str:
        """
        提交任务：按剩余时间分流——
        * 距触发 > 7 天  → 磁盘缓存（扫描任务提前检测回捞）
        * 距触发 <= 7 天 → 四层时间轮
        * 已到期          → 立即调度执行
        """
        if not isinstance(task, BaseTask):
            raise TypeError("submit 需要 BaseTask 子类实例")
        with self._lock:
            self.tasks[task.task_id] = task
            self._arm(task)
        return task.task_id

    def _arm(self, task: BaseTask) -> None:
        """把任务放入磁盘缓存或时间轮（submit / 周期重入 / 扫描回捞共用）。"""
        delay = task.run_at - time.time() if task.run_at is not None else 0
        if delay > self.horizon:
            self.disk.save(task)               # > 7 天：落盘
        else:
            self.wheel.add(task)               # <= 7 天：时间轮（过期则立即触发）

    def cancel(self, task_id: str) -> bool:
        """取消任务：优先从时间轮/磁盘移除；运行中则置位 stop_event 协作退出。"""
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return False
            was_running = task.state is TaskState.RUNNING
            removed = self.wheel.cancel(task_id)
            if not removed:
                # 磁盘缓存中的任务：直接删除文件
                self.disk.delete(task_id)
                removed = True
            task.cancel()
            # 未运行的任务在此触发 cancel 钩子；运行中的由 execute() 统一触发
            if not was_running:
                ctx = TaskContext(task, task.stop_event, self.semaphore)
                task.on_cancel(ctx)
            return removed

    def get_task(self, task_id: str) -> Optional[BaseTask]:
        return self.tasks.get(task_id)

    # ------------------------------------------------------------------
    # 执行调度（时间轮到期回调）
    # ------------------------------------------------------------------
    def _on_due(self, task: BaseTask) -> None:
        """时间轮到期：提交到 runner 池执行（并发受信号量约束）。"""
        self.runner_pool.submit(self._run_task, task)

    def _run_task(self, task: BaseTask) -> None:
        """工作线程中执行一次任务；周期任务执行完后重新计算并重入队。"""
        ctx = task.execute(self.executor, self.semaphore)
        if task.state is TaskState.CANCELLED:
            return
        try:
            nxt = task.next_run_at(time.time())
        except Exception:
            logger.exception("任务 %s 计算下次触发时间失败，不再调度", task.task_id)
            return
        if nxt is None:
            return                                   # 一次性任务：结束
        with self._lock:
            task.reset()
            task.run_at = nxt
            self._arm(task)                          # 周期任务：按新时间重新入队

    # ------------------------------------------------------------------
    # 磁盘扫描：提前一天检测远期任务
    # ------------------------------------------------------------------
    def _scan_loop(self) -> None:
        while self._running:
            time.sleep(self.scan_interval)
            try:
                self._scan_once()
            except Exception:
                logger.exception("磁盘扫描异常")

    def _scan_once(self) -> int:
        """扫描磁盘缓存，把进入 7 天窗口的任务加载回内存时间轮。"""
        due = self.disk.scan_due(self.horizon)
        for task in due:
            with self._lock:
                self.tasks.setdefault(task.task_id, task)
                self._arm(task)
        if due:
            logger.info("磁盘扫描回捞 %d 个任务进入内存时间轮", len(due))
        return len(due)

    # ------------------------------------------------------------------
    # 持久化：内存任务落盘 / 重启还原
    # ------------------------------------------------------------------
    def save_state(self) -> int:
        """把内存中 PENDING/RUNNING 任务以 JSON 元信息落盘。"""
        return self.store.save(self.tasks)

    def load_state(self) -> int:
        """重启后加载：还原任务并重新入队（到期任务会立即触发）。"""
        tasks = self.store.restore_all()
        for task in tasks:
            self.submit(task)
        if tasks:
            logger.info("重启还原 %d 个内存任务", len(tasks))
        return len(tasks)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, int]:
        """按状态统计（运维/展示用）。"""
        stats: Dict[str, int] = {}
        with self._lock:
            for task in self.tasks.values():
                key = task.state.value
                stats[key] = stats.get(key, 0) + 1
        stats["wheel_pending"] = self.wheel.count()
        stats["disk_cached"] = len(self.disk.list_index())
        return stats


__all__ = [
    "Scheduler",
    "BaseTask",
    "TaskState",
    "TaskContext",
    "ScheduledTask",
    "PeriodicTask",
    "DelayTask",
    "ScheduledPeriodicTask",
    "DelayPeriodicTask",
    "HierarchicalTimeWheel",
    "DiskTaskStore",
    "TaskStore",
    "restore_task",
    "DEFAULT_HORIZON_DAYS",
]
