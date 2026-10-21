# -*- coding: utf-8 -*-
"""
情感任务 EmotionTask
====================
将情感引擎的衰减逻辑接入调度框架，作为周期任务定期执行 ``engine.decay()``。

示例::

    from com.robot.scheduler import Scheduler
    from com.robot.emotion import EmotionEngine, EmotionTask

    engine = EmotionEngine()
    task = EmotionTask(engine=engine, interval=1)   # 每秒衰减一次

    sched = Scheduler()
    sched.start()
    sched.submit(task)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from ..scheduler.base import BaseTask, TaskState, parse_timeout

logger = logging.getLogger(__name__)


class EmotionTask(BaseTask):
    """
    情感衰减周期任务。

    将 ``EmotionEngine.decay()`` 包装为调度器可管理的周期任务，
    每个 interval 秒执行一次衰减，使情感状态逐步回归中性。

    Parameters
    ----------
    engine       : 情感引擎实例（必须）
    interval     : 衰减间隔秒数，默认 1
    decay_factor : 可选，覆盖引擎的衰减系数
    timeout      : 单次执行超时
    hooks        : 生命周期钩子
    task_id      : 任务 ID
    """

    is_periodic = True

    def __init__(
        self,
        engine: Any,
        interval: float = 1.0,
        decay_factor: Optional[float] = None,
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval 必须大于 0")

        super().__init__(
            name="emotion_decay",
            callback=None,
            timeout=parse_timeout(timeout),
            hooks=hooks,
            task_id=task_id,
        )
        self.engine = engine
        self.interval = float(interval)
        if decay_factor is not None:
            self.engine.decay_factor = decay_factor
        self.run_at = self.created_at + self.interval

    # ------------------------------------------------------------------
    # 调度接口
    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        if self.state is TaskState.CANCELLED:
            return None
        now = now if now is not None else self.created_at
        nxt = self.run_at + self.interval
        while nxt <= now:
            nxt += self.interval
        return nxt

    def execute(self, executor, semaphore) -> Any:
        """覆写执行逻辑：直接调用引擎衰减，不走线程池回调。"""
        from ..scheduler.base import TaskContext

        ctx = TaskContext(self, self.stop_event, semaphore)
        ctx.start_time = __import__("time").time()
        self.on_start(ctx)

        try:
            self.engine.decay()
            ctx.result = self.engine.state.copy()
            self.state = TaskState.SUCCESS
        except Exception as exc:
            ctx.error = exc
            self.state = TaskState.FAILED
            self.on_error(ctx)
            logger.exception("情感衰减任务执行失败")
        finally:
            ctx.end_time = __import__("time").time()
            self.on_end(ctx)

        return ctx

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["interval"] = self.interval
        data["engine"] = self.engine.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmotionTask:
        from .engine import EmotionEngine

        engine = EmotionEngine.from_dict(data.get("engine", {}))
        task = cls(
            engine=engine,
            interval=data.get("interval", 1.0),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<EmotionTask interval={self.interval}s "
            f"emotion={self.engine.emotion.value} "
            f"state={self.state.value}>"
        )
