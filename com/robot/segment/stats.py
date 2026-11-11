# -*- coding: utf-8 -*-
"""滑动统计与运行指标。

对应设计文档 §2「滑动窗口 2 秒」的**读法 A**（统计窗，不是触发窗，文档 §7-G4）
与 §8 的可观测性要求。没有指标就无法判断 G1/G10/G12 的补全是否真的更好。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .models import CloseReason, SegState, Segment


class SegmentStats:
    """噪声底噪估计 + 分段质量指标。

    所有方法都在 VAD 线程内调用，**不做跨线程加锁**；:meth:`report` 只读快照。
    """

    def __init__(self, window_ms: int = 2000, noise_percentile: float = 0.2, fragment_threshold_ms: int = 450) -> None:
        self.window_ms = window_ms
        self.noise_percentile = noise_percentile
        self.fragment_threshold_ms = fragment_threshold_ms
        self._frames: Deque[Tuple[int, float, float]] = deque()   # (end_ms, level_db, prob)
        self.total_frames = 0
        self.speech_frames = 0
        self.gap_frames = 0
        self.gap_ms = 0
        self.onset_latencies: List[int] = []
        self.offset_latencies: List[int] = []
        self.final_segments = 0
        self.short_final_segments = 0
        self.partial_segments = 0
        self.discarded_short = 0
        self.gate_blocked = 0
        self.close_counts: Dict[str, int] = {r.value: 0 for r in CloseReason}
        self.state_flips = 0

    # ------------------------------------------------------------------ 统计窗
    def push_frame(self, end_ms: int, level_db: float, prob: float) -> None:
        self._frames.append((end_ms, level_db, prob))
        self.total_frames += 1
        if prob >= 0.5:
            self.speech_frames += 1
        cutoff = end_ms - self.window_ms
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def noise_floor_db(self, default: float = -120.0, min_samples: int = 8) -> float:
        """**非语音帧**电平的低分位数，作为自适应底噪（文档 §7-G10）。

        只用 ``prob < 0.5`` 的帧：整窗都是语音时，如果按全部帧取分位数，
        「底噪」会等于语音电平，门限 ``底噪 + 6dB`` 便永远达不到，
        于是正常说话被判低能量丢弃（实测会让截断后的续段直接消失）。
        窗内非语音帧太少时返回 ``default``，表示底噪未收敛。
        """
        levels = sorted(f[1] for f in self._frames if f[2] < 0.5)
        if len(levels) < min_samples:
            return default
        idx = max(0, min(len(levels) - 1, int(len(levels) * self.noise_percentile)))
        return levels[idx]

    # ------------------------------------------------------------------ 事件
    def note_onset(self, start_ms: int, decision_ms: int) -> None:
        self.onset_latencies.append(decision_ms - start_ms)

    def note_offset(self, last_speech_ms: int, decision_ms: int) -> None:
        self.offset_latencies.append(decision_ms - last_speech_ms)

    def note_close(self, seg: Segment) -> None:
        if seg.partial:
            self.partial_segments += 1
        else:
            self.final_segments += 1
            if seg.duration_ms < self.fragment_threshold_ms:
                self.short_final_segments += 1
            self.close_counts[seg.close_reason.value] = self.close_counts.get(seg.close_reason.value, 0) + 1

    def note_discard_short(self) -> None:
        self.discarded_short += 1

    def note_gate_blocked(self) -> None:
        self.gate_blocked += 1

    def note_gap(self, ms: int) -> None:
        self.gap_frames += 1
        self.gap_ms += ms

    def note_state_flip(self, old: SegState, new: SegState) -> None:  # noqa: ARG002
        self.state_flips += 1

    # ------------------------------------------------------------------ 汇总
    @staticmethod
    def _pct(values: List[int], q: float) -> Optional[int]:
        if not values:
            return None
        s = sorted(values)
        return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]

    def report(self) -> Dict[str, object]:
        total_final = max(1, self.final_segments)
        return {
            "frames": self.total_frames,
            "speech_ratio": round(self.speech_frames / max(1, self.total_frames), 3),
            "final_segments": self.final_segments,
            "short_final_segments": self.short_final_segments,
            "partial_segments": self.partial_segments,
            "discarded_short": self.discarded_short,
            "gate_blocked": self.gate_blocked,
            "state_flips": self.state_flips,
            "fragmentation_rate": round(self.short_final_segments / total_final, 3),
            "close_counts": dict(self.close_counts),
            "onset_latency_p50": self._pct(self.onset_latencies, 0.50),
            "onset_latency_p95": self._pct(self.onset_latencies, 0.95),
            "offset_latency_p50": self._pct(self.offset_latencies, 0.50),
            "offset_latency_p95": self._pct(self.offset_latencies, 0.95),
            "gap_frames": self.gap_frames,
            "gap_ms": self.gap_ms,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"SegmentStats({self.report()})"
