# -*- coding: utf-8 -*-
"""
周期任务类 PeriodicTask
=======================
按**固定时间间隔**重复执行的回调任务。

示例::

    task = PeriodicTask(
        name="心跳上报",
        callback=my_cb,
        interval=10,          # 每 10 秒执行一次
        start_after=5,        # 首次执行在提交后 5 秒（默认等于 interval）
        timeout=3,
        hooks={"start": s, "end": e, "timeout": t},
    )
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BaseTask, TaskState, parse_timeout


class PeriodicTask(BaseTask):
    """周期任务：间隔固定秒数重复执行。"""

    is_periodic = True

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        interval: float,
        start_after: Optional[float] = None,
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval 必须大于 0")
        super().__init__(name, callback, timeout=parse_timeout(timeout),
                         hooks=hooks, task_id=task_id)
        self.interval = float(interval)
        self.start_after = float(start_after if start_after is not None else interval)
        self.run_at = self.created_at + self.start_after

    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        if self.state is TaskState.CANCELLED or self.run_at is None:
            return None
        now = now if now is not None else self.created_at
        # 下一次 = 上一次触发时刻 + interval（若已过期则顺延到未来）
        nxt = self.run_at + self.interval
        while nxt <= now:
            nxt += self.interval
        return nxt

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["interval"] = self.interval
        data["start_after"] = self.start_after
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeriodicTask":
        task = cls(
            name=data["name"],
            callback=cls._restore_callable(data.get("callback")),
            interval=data.get("interval", 1),
            start_after=data.get("start_after"),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task
