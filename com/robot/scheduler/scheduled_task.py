# -*- coding: utf-8 -*-
"""
定时任务类 ScheduledTask
========================
在指定的**绝对时间点**执行一次的回调任务（一次性）。

示例::

    task = ScheduledTask(
        name="开机自检",
        callback=my_cb,                 # def my_cb(ctx): ...
        run_at="2026-10-10 12:00:00",   # datetime / 时间戳 / 字符串均可
        timeout=30,                     # 30 秒超时，超时置位 ctx.stop_event
        hooks={"start": s, "end": e, "timeout": t},
    )
    scheduler.submit(task)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BaseTask, TaskState, to_epoch, parse_timeout


class ScheduledTask(BaseTask):
    """定时任务：到点执行一次。"""

    is_periodic = False

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        run_at: Any,
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        super().__init__(name, callback, timeout=parse_timeout(timeout),
                         hooks=hooks, task_id=task_id)
        self.run_at = to_epoch(run_at)

    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        # 一次性任务：触发过（终态）后不再调度
        if self._fired or self.state is TaskState.CANCELLED:
            return None
        return self.run_at

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["run_at"] = self.run_at
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        task = cls(
            name=data["name"],
            callback=cls._restore_callable(data.get("callback")),
            run_at=data.get("run_at"),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task
