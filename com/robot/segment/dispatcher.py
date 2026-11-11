# -*- coding: utf-8 -*-
"""段派发。

对应设计文档 §5.6 与 §7-G11。核心约定：

* ``D``（消费游标）在**深拷贝完成后立即前进**，绝不等待 ASR 结果；
* 投递队列有界，满时优先丢 ``partial``、保留 ``final``，并计数上报；
* :class:`Segment` 携带脱离环形缓冲的 ``samples``，下游可安全持有、可重放。
"""
from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .models import Segment


class ISegmentDispatcher(ABC):
    """段派发出口。"""

    @abstractmethod
    def dispatch(self, seg: Segment) -> bool:
        """非阻塞投递；返回 ``False`` 表示队列满且无可丢弃对象。"""


class QueueDispatcher(ISegmentDispatcher):
    """有界队列派发器（ASR 线程用 :meth:`get` 消费）。"""

    def __init__(self, maxsize: int = 4, drop_policy: str = "drop_partial") -> None:
        if drop_policy not in ("drop_partial", "reject"):
            raise ValueError(f"未知 drop_policy: {drop_policy!r}")
        self._q: "queue.Queue[Segment]" = queue.Queue(maxsize=max(1, maxsize))
        self._policy = drop_policy
        self._lock = threading.Lock()
        self.dropped = 0
        self.delivered = 0

    # ------------------------------------------------------------------
    def dispatch(self, seg: Segment) -> bool:
        if self._try_put(seg):
            return True
        if self._policy == "reject":
            self._bump_dropped()
            return False
        # 队列满：挤掉一条最旧的 partial 给新段让位（final 永不主动丢弃）
        if self._evict_partial() and self._try_put(seg):
            return True
        self._bump_dropped()
        return False

    def _try_put(self, seg: Segment) -> bool:
        try:
            self._q.put_nowait(seg)
        except queue.Full:
            return False
        with self._lock:
            self.delivered += 1
        return True

    def _bump_dropped(self) -> None:
        with self._lock:
            self.dropped += 1

    def _evict_partial(self) -> bool:
        """从队列中取出最旧的一条 partial；成功返回 True。"""
        evicted: List[Segment] = []
        found = False
        with self._lock:
            while True:
                try:
                    s = self._q.get_nowait()
                except queue.Empty:
                    break
                if not found and not s.is_final:
                    found = True
                    self.dropped += 1
                    continue
                evicted.append(s)
            for s in evicted:
                self._q.put_nowait(s)
        return found

    # ------------------------------------------------------------------
    def get(self, timeout: Optional[float] = None) -> Optional[Segment]:
        try:
            return self._q.get(timeout=timeout) if timeout is not None else self._q.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> List[Segment]:
        out: List[Segment] = []
        while True:
            s = self.get()
            if s is None:
                return out
            out.append(s)

    def qsize(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict:
        with self._lock:
            return {"delivered": self.delivered, "dropped": self.dropped, "qsize": self._q.qsize()}


class CallbackDispatcher(ISegmentDispatcher):
    """直接回调（同线程）派发器，用于测试与轻量集成。"""

    def __init__(self, sink: Callable[[Segment], None]) -> None:
        self._sink = sink
        self.segments: List[Segment] = []

    def dispatch(self, seg: Segment) -> bool:
        self.segments.append(seg)
        self._sink(seg)
        return True
