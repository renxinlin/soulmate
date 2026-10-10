# -*- coding: utf-8 -*-
"""
延时任务类 DelayTask
====================
提交后经过 **delay 秒** 延迟，执行**一次**的回调任务。

示例::

    task = DelayTask(
        name="延迟发消息",
        callback=my_cb,
        delay=3,               # 3 秒后执行一次
        timeout=2,             # 执行超时 2 秒
        hooks={"start": s, "end": e, "timeout": t},
    )
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BaseTask, TaskState, parse_timeout


class DelayTask(BaseTask):
    """延时任务：延迟固定秒数后执行一次。"""

    is_periodic = False

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        delay: float,
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if delay < 0:
            raise ValueError("delay 不能为负数")
        super().__init__(name, callback, timeout=parse_timeout(timeout),
                         hooks=hooks, task_id=task_id)
        self.delay = float(delay)
        self.run_at = self.created_at + self.delay

    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        if self._fired or self.state is TaskState.CANCELLED:
            return None
        return self.run_at

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["delay"] = self.delay
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelayTask":
        task = cls(
            name=data["name"],
            callback=cls._restore_callable(data.get("callback")),
            delay=data.get("delay", 0),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task
