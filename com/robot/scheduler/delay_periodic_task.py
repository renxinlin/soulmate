# -*- coding: utf-8 -*-
"""
延时周期任务类 DelayPeriodicTask
================================
先延迟 **delay 秒** 执行第一次，之后每隔 **interval 秒** 重复执行。

示例::

    task = DelayPeriodicTask(
        name="延迟启动的心跳",
        callback=my_cb,
        delay=5,        # 5 秒后第一次执行
        interval=10,    # 之后每 10 秒一次
        timeout=3,
        hooks={"start": s, "end": e, "timeout": t},
    )
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BaseTask, TaskState, parse_timeout


class DelayPeriodicTask(BaseTask):
    """延时周期任务：delay 后首次执行，之后按 interval 周期执行。"""

    is_periodic = True

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        delay: float,
        interval: float,
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if delay < 0:
            raise ValueError("delay 不能为负数")
        if interval <= 0:
            raise ValueError("interval 必须大于 0")
        super().__init__(name, callback, timeout=parse_timeout(timeout),
                         hooks=hooks, task_id=task_id)
        self.delay = float(delay)
        self.interval = float(interval)
        self.run_at = self.created_at + self.delay

    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        if self.state is TaskState.CANCELLED or self.run_at is None:
            return None
        now = now if now is not None else self.created_at
        nxt = self.run_at + self.interval
        while nxt <= now:
            nxt += self.interval
        return nxt

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["delay"] = self.delay
        data["interval"] = self.interval
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelayPeriodicTask":
        task = cls(
            name=data["name"],
            callback=cls._restore_callable(data.get("callback")),
            delay=data.get("delay", 0),
            interval=data.get("interval", 1),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task
