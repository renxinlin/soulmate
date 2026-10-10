# -*- coding: utf-8 -*-
"""
定时周期任务类 ScheduledPeriodicTask
====================================
按 **cron 表达式**（分 时 日 月 周，5 段）在固定的墙钟时刻重复执行。

表达式支持：``*``、``*/n``、``a-b``、``a,b,c`` 及其组合。

示例::

    # 每天 08:30 执行
    task = ScheduledPeriodicTask(name="晨会提醒", callback=cb, cron="30 8 * * *")
    # 每周一、三的 10:00 执行
    task = ScheduledPeriodicTask(name="巡检", callback=cb, cron="0 10 * * 1,3")
    # 每小时的第 15 分钟执行
    task = ScheduledPeriodicTask(name="上报", callback=cb, cron="15 * * * *")

cron 字段取值：分(0-59) 时(0-23) 日(1-31) 月(1-12) 周(0-6, 0=周日)。
日与周同时受限时按标准 cron 语义：**两者任一匹配**即可。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Set

from .base import BaseTask, TaskState, parse_timeout

_FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


def _parse_field(spec: str, lo: int, hi: int) -> Optional[Set[int]]:
    """解析单个 cron 字段；'*' 返回 None 表示不限制。"""
    spec = spec.strip()
    if spec == "*":
        return None
    result: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            base, _, step_s = part.partition("/")
            step = int(step_s)
            if base == "*":
                result.update(range(lo, hi + 1, step))
                continue
            part = base
        if "-" in part:
            a, _, b = part.partition("-")
            start, end = int(a), int(b)
            result.update(range(start, end + 1, step))
        else:
            result.add(int(part))
    return result if result else None


class _CronSpec:
    """cron 五段式规格。"""

    __slots__ = ("minute", "hour", "dom", "month", "dow")

    def __init__(self, minute, hour, dom, month, dow) -> None:
        self.minute = minute
        self.hour = hour
        self.dom = dom
        self.month = month
        self.dow = dow

    @classmethod
    def parse(cls, expr: str) -> "_CronSpec":
        parts = expr.split()
        if len(parts) != 5:
            raise ValueError(f"cron 表达式必须为 5 段(分 时 日 月 周): {expr!r}")
        fields = [_parse_field(p, lo, hi)
                  for p, (lo, hi) in zip(parts, _FIELD_RANGES)]
        return cls(*fields)

    def match(self, dt: datetime) -> bool:
        if self.minute is not None and dt.minute not in self.minute:
            return False
        if self.hour is not None and dt.hour not in self.hour:
            return False
        if self.month is not None and dt.month not in self.month:
            return False
        if self.dom is not None or self.dow is not None:
            dom_ok = self.dom is None or dt.day in self.dom
            dow_ok = self.dow is None or dt.weekday() in self.dow
            if not (dom_ok or dow_ok):          # 标准 cron：任一匹配即可
                return False
        return True

    def next_match(self, now_ts: float) -> float:
        """计算 now 之后的下一个匹配时刻（秒级截断到分）。"""
        cur = datetime.fromtimestamp(now_ts).replace(second=0, microsecond=0) \
            + timedelta(minutes=1)
        for _ in range(366 * 4):                # 最多向前找 4 年
            if self.month is not None and cur.month not in self.month:
                cur = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
                cur = cur.replace(hour=0, minute=0)
                continue
            if self.dom is not None or self.dow is not None:
                dom_ok = self.dom is None or cur.day in self.dom
                dow_ok = self.dow is None or cur.weekday() in self.dow
                if not (dom_ok or dow_ok):
                    cur = (cur + timedelta(days=1)).replace(hour=0, minute=0)
                    continue
            if self.hour is not None and cur.hour not in self.hour:
                cur = (cur + timedelta(hours=1)).replace(minute=0)
                continue
            if self.minute is not None and cur.minute not in self.minute:
                cur = cur + timedelta(minutes=1)
                continue
            return cur.timestamp()
        raise ValueError(f"cron 表达式在 4 年内无匹配: {self!r}")


class ScheduledPeriodicTask(BaseTask):
    """定时周期任务：按 cron 表达式在固定墙钟时刻重复执行。"""

    is_periodic = True

    def __init__(
        self,
        name: str,
        callback: Optional[Callable],
        cron: str = "* * * * *",
        timeout: Optional[Any] = None,
        hooks: Optional[Dict[str, Callable]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        super().__init__(name, callback, timeout=parse_timeout(timeout),
                         hooks=hooks, task_id=task_id)
        self.cron = cron
        self._spec = _CronSpec.parse(cron)
        self.run_at = self._spec.next_match(time.time())

    # ------------------------------------------------------------------
    def next_run_at(self, now: Optional[float] = None) -> Optional[float]:
        if self.state is TaskState.CANCELLED:
            return None
        return self._spec.next_match(now if now is not None else time.time())

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["cron"] = self.cron
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledPeriodicTask":
        task = cls(
            name=data["name"],
            callback=cls._restore_callable(data.get("callback")),
            cron=data.get("cron", "* * * * *"),
            timeout=data.get("timeout"),
            hooks={k: cls._restore_callable(v)
                   for k, v in (data.get("hooks") or {}).items()},
            task_id=data.get("task_id"),
        )
        task._restore_state(data)
        return task
