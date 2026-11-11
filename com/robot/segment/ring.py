# -*- coding: utf-8 -*-
"""音频环形缓冲。

对应设计文档 §3.1。要点：

* 容量取 2 的幂（样本数），索引可用掩码取模；
* 时间坐标是「单调毫秒」，不是数组下标（文档 §2）；
* 单写单读：采集线程 :meth:`write` 推进 ``W``，VAD 线程只读；
* 被 ``W - capacity`` 跨过的数据不可恢复，读取时显式返回 :class:`SampleStatus`
  而不是静默返回错数据（文档 §7-G12）。
"""
from __future__ import annotations

from typing import Tuple

from .config import SegmentConfig
from .models import SampleStatus


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class AudioRingBuffer:
    """int16 / mono PCM 环形缓冲。"""

    def __init__(self, cfg: SegmentConfig) -> None:
        self.cfg = cfg
        self.samples_per_ms = cfg.sample_rate // 1000
        self.bytes_per_ms = cfg.bytes_per_ms
        cap_samples = _next_pow2(max(1, cfg.ring_capacity_ms * self.samples_per_ms))
        self._buf = bytearray(cap_samples * 2)
        self._mask = cap_samples - 1
        self._cap_samples = cap_samples
        self.capacity_ms = cap_samples // self.samples_per_ms
        self._w_ms = 0          # 已提交音频的结束时刻
        self.dropped_ms = 0     # 因写入超过容量而被覆盖的量

    # ------------------------------------------------------------------
    @property
    def w_ms(self) -> int:
        """写指针（最新提交帧的结束时刻）。"""
        return self._w_ms

    @property
    def oldest_ms(self) -> int:
        """仍可读取的最早时刻（可能是负数 → 表示尚未填满）。"""
        return self._w_ms - self.capacity_ms

    def space_left_ms(self, cursor_ms: int) -> int:
        """从某个游标（通常是 ``D``）到写指针之间还可安全保留的毫秒数。"""
        return self.capacity_ms - (self._w_ms - cursor_ms)

    # ------------------------------------------------------------------
    def write(self, data: bytes, start_ms: int) -> int:
        """写入 ``data``（int16 mono），返回**这一帧里留不下的毫秒数**。

        ``start_ms`` 必须等于 ``self.w_ms``（顺序写入）；否则视为采集时间轴断裂，
        调用方应把该帧标记为 ``discontinuous``。
        """
        if len(data) % 2:
            raise ValueError("PCM 长度必须为偶数字节（int16）")
        n = len(data) // 2
        if n == 0:
            return 0
        data_ms = n // self.samples_per_ms
        end_ms = start_ms + data_ms
        if end_ms <= self._w_ms:
            return 0                      # 迟到的旧帧（整帧都在写指针之前），忽略
        # 写完只剩 [end_ms - capacity, end_ms) 还读得到，之前的部分就是丢弃量：
        # 既包含「落在可保留窗口之前、根本没写进去的」，也包含「单帧比缓冲还长、
        # 开头立刻被自己盖掉的」。显式上报而不是静默覆盖（文档 §7-G12）。
        dropped = max(0, end_ms - self.capacity_ms - start_ms)
        min_start = max(0, self._w_ms - self.capacity_ms)
        if start_ms < min_start:
            data = data[(min_start - start_ms) * self.samples_per_ms * 2:]
            start_ms = min_start
        self.dropped_ms += dropped
        self._put(start_ms, data)
        self._w_ms = end_ms
        return dropped

    # ------------------------------------------------------------------
    def read(self, start_ms: int, end_ms: int) -> Tuple[bytes, SampleStatus]:
        """读取 ``[start_ms, end_ms)``。区间越界时补零并返回相应 :class:`SampleStatus`。"""
        if end_ms <= start_ms:
            return b"", SampleStatus.CLEAN
        status = SampleStatus.CLEAN
        oldest = self.oldest_ms
        if start_ms < oldest:
            status = SampleStatus.CLAMPED
        if end_ms > self._w_ms:
            status = SampleStatus.CLAMPED

        s = max(start_ms, max(oldest, 0))
        e = min(end_ms, self._w_ms)
        if e <= s:
            return b"\x00" * ((end_ms - start_ms) * self.bytes_per_ms), status

        body = self._get(s, e)
        pad_head = (s - start_ms) * self.bytes_per_ms if s > start_ms else 0
        pad_tail = (end_ms - e) * self.bytes_per_ms if e < end_ms else 0
        if pad_head or pad_tail:
            status = SampleStatus.CLAMPED
            body = b"\x00" * pad_head + body + b"\x00" * pad_tail
        return body, status

    # ------------------------------------------------------------------
    def _put(self, start_ms: int, data: bytes) -> None:
        s0 = start_ms * self.samples_per_ms
        n = len(data) // 2
        if n > self._cap_samples:
            # 单帧比整个缓冲还长：只有末尾 capacity 个样本留得下来，
            # 否则回绕段会越界、把 bytearray 撑大（掩码随之失效）
            data = data[-self._cap_samples * 2:]
            s0 += n - self._cap_samples
            n = self._cap_samples
        i0 = s0 & self._mask
        first = min(n, self._cap_samples - i0)
        b0 = self._buf
        b0[i0 * 2:(i0 + first) * 2] = data[:first * 2]
        rest = n - first
        if rest > 0:
            b0[0:rest * 2] = data[first * 2:]

    def _get(self, start_ms: int, end_ms: int) -> bytes:
        s0 = start_ms * self.samples_per_ms
        n = (end_ms - start_ms) * self.samples_per_ms
        i0 = s0 & self._mask
        first = min(n, self._cap_samples - i0)
        b0 = self._buf
        out = bytes(b0[i0 * 2:(i0 + first) * 2])
        if n - first > 0:
            out += bytes(b0[0:(n - first) * 2])
        return out
