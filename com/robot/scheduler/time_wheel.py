# -*- coding: utf-8 -*-
"""
四层时间轮 HierarchicalTimeWheel（天 / 时 / 分 / 秒）
====================================================
经典级联式分层时间轮：

* 秒轮：60 槽 × 1 秒
* 分轮：60 槽 × 1 分
* 时轮：24 槽 × 1 时
* 天轮：366 槽 × 1 天

插入规则：取 ``unit <= delay`` 的**最大**层（分/时/天/秒的经典分层规则），
把任务放入 ``(cursor + ticks) % size`` 槽位；``ticks = delay // unit``（至少 1）。

推进规则：ticker 线程每 ``tick_interval`` 秒调用 ``advance()``，秒轮每 tick
前进 1 槽并取出该槽任务：到期（run_at <= now）的直接回调 ``on_due``，
未到期的重新按剩余延迟下插到更小层；分/时/天轮按**真实时间**（每 60s /
3600s / 86400s）级联前进 1 槽，不依赖秒轮回绕，因此 tick 频率可低于 1 秒。

单个槽用 ``deque`` 存放（同一时刻多个任务）。``_entries`` 维护
``task_id -> (layer, idx)`` 用于 O(1) 取消。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .base import BaseTask

logger = logging.getLogger(__name__)

# 天轮槽数：366（含闰年余量；>7 天的任务由调度器走磁盘缓存，不会触及该上限）
_DAY_SLOTS = 366


class _Layer:
    __slots__ = ("unit", "size", "slots", "cursor")

    def __init__(self, unit: float, size: int) -> None:
        self.unit = unit                # 单槽对应的秒数
        self.size = size                # 槽数
        self.slots: List[Deque[BaseTask]] = [deque() for _ in range(size)]
        self.cursor = 0                 # 当前指针

    @property
    def span(self) -> float:
        """本层能覆盖的总时间跨度（秒）。"""
        return self.unit * self.size


class HierarchicalTimeWheel:
    """四层（天/时/分/秒）级联时间轮，处理 7 天以内的任务。"""

    def __init__(self, on_due: Optional[Callable[[BaseTask], None]] = None,
                 tick_interval: float = 1.0) -> None:
        self.days = _Layer(86400, _DAY_SLOTS)
        self.hours = _Layer(3600, 24)
        self.minutes = _Layer(60, 60)
        self.seconds = _Layer(1, 60)
        # 从小到大排列（unit 递增）：插入时取 unit <= delay 的最大层
        self._bottom_up = (self.seconds, self.minutes, self.hours, self.days)

        self._on_due = on_due
        self._tick_interval = tick_interval
        self._lock = threading.RLock()
        self._entries: Dict[str, Tuple[_Layer, int]] = {}
        self._running = False
        self._ticker: Optional[threading.Thread] = None
        now = time.time()
        # 上层按真实时间级联（分/时/天），不依赖秒轮回绕
        self._cascade_ts = {
            self.minutes: now + 60.0,
            self.hours: now + 3600.0,
            self.days: now + 86400.0,
        }

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def add(self, task: BaseTask) -> None:
        """把任务加入时间轮；若已到期则立即触发。"""
        run_at = task.run_at
        if run_at is None:
            raise ValueError(f"任务 {task.name} 缺少 run_at")
        now = time.time()
        with self._lock:
            if run_at <= now:
                self._fire(task)
                return
            self._insert(task, run_at - now)

    def cancel(self, task_id: str) -> bool:
        """从时间轮中移除任务；返回是否移除成功。"""
        with self._lock:
            entry = self._entries.pop(task_id, None)
            if entry is None:
                return False
            layer, idx = entry
            try:
                layer.slots[idx].remove(
                    next(t for t in layer.slots[idx] if t.task_id == task_id))
            except (StopIteration, ValueError):
                return False
            return True

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def __len__(self) -> int:
        return self.count()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _insert(self, task: BaseTask, delay: float) -> None:
        """
        层级选择：取 ``unit <= delay`` 的**最大**层（经典分层时间轮规则）。
        * delay < 1s     → 秒轮（槽位 +1，下一 tick 触发）
        * 60s <= d < 1h  → 分轮
        * 1h <= d < 1d   → 时轮
        * 1d <= d        → 天轮
        """
        chosen = self.seconds
        for layer in self._bottom_up:
            if layer.unit <= delay:
                chosen = layer
            else:
                break
        ticks = max(1, int(delay // chosen.unit))
        if ticks >= chosen.size:
            ticks = chosen.size - 1                 # 极端兜底：放最后一格
        idx = (chosen.cursor + ticks) % chosen.size
        chosen.slots[idx].append(task)
        self._entries[task.task_id] = (chosen, idx)

    def _fire(self, task: BaseTask) -> None:
        """任务到期：从索引中移除并回调调度器。"""
        self._entries.pop(task.task_id, None)
        if self._on_due is not None:
            try:
                self._on_due(task)
            except Exception:
                logger.exception("时间轮触发任务 %s 失败", task.task_id)

    def advance(self) -> None:
        """ticker 周期调用：秒轮逐槽推进；分/时/天轮按真实时间级联推进。"""
        now = time.time()
        with self._lock:
            self._advance(self.seconds)
            if now >= self._cascade_ts[self.minutes]:
                self._advance(self.minutes)
                self._cascade_ts[self.minutes] = now + 60.0
            if now >= self._cascade_ts[self.hours]:
                self._advance(self.hours)
                self._cascade_ts[self.hours] = now + 3600.0
            if now >= self._cascade_ts[self.days]:
                self._advance(self.days)
                self._cascade_ts[self.days] = now + 86400.0

    def _advance(self, layer: _Layer) -> None:
        layer.cursor = (layer.cursor + 1) % layer.size
        slot = layer.slots[layer.cursor]
        now = time.time()
        tasks = list(slot)
        slot.clear()
        for task in tasks:
            self._entries.pop(task.task_id, None)
            if task.run_at <= now + 0.05:        # 允许 50ms 时钟误差
                self._fire(task)
            else:
                self._insert(task, task.run_at - now)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ticker = threading.Thread(
            target=self._tick_loop, name="time-wheel-ticker", daemon=True)
        self._ticker.start()

    def _tick_loop(self) -> None:
        while self._running:
            time.sleep(self._tick_interval)
            try:
                self.advance()
            except Exception:
                logger.exception("时间轮 tick 异常")

    def stop(self) -> None:
        self._running = False
        if self._ticker is not None:
            self._ticker.join(timeout=2)
            self._ticker = None
