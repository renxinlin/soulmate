# -*- coding: utf-8 -*-
"""分段模块数据结构（DTO）。

对应设计文档 §3.2。与 ``docs/speaker_pipeline_design.md`` 中的 ``AudioFrame`` /
``AudioSegment`` 保持字段命名一致，便于后续复用（文档 §7-G17）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class SegState(Enum):
    """主状态机状态——按需求只保留两个状态（R3）。"""

    IDLE = "idle"
    SPEAKING = "speaking"


class CloseReason(Enum):
    """段闭合原因——下游据此决定是否做同句拼接。"""

    SILENCE = "silence"              # R4：静音超时，正常闭合
    MAX_DURATION = "max_duration"    # R5：时长上限硬截断，句子未必说完
    RING_PRESSURE = "ring"           # I2：环形缓冲压力，保护性闭合
    GATE_CLOSED = "gate"             # 门控关闭（未唤醒 / 开始 TTS）
    EOS = "eos"                      # 采集停止 / 会话结束


class SampleStatus(Enum):
    """段内音频完整性。"""

    CLEAN = "clean"
    GAP = "gap"        # 含 discontinuous 帧（采集 overrun）
    CLAMPED = "clamped"  # 请求区间已被环形缓冲覆盖，缺的部分补零


@dataclass
class AudioFrame:
    """一帧音频及其 VAD 判决结果。

    ``ts_start_ms`` / ``ts_end_ms`` 均为「自采集启动的单调毫秒」，
    **不使用环形数组下标作为时间坐标**（文档 §2）。
    """

    ts_start_ms: int
    ts_end_ms: int
    prob: float = 0.0
    level_db: float = -120.0
    samples: Optional[bytes] = None
    discontinuous: bool = False
    seq: int = 0

    @property
    def duration_ms(self) -> int:
        return self.ts_end_ms - self.ts_start_ms

    @property
    def is_speech(self) -> bool:
        """便捷只读：由 ``prob`` 推断，判决阈值本身在引擎内。"""
        return self.prob >= 0.5


@dataclass
class Candidate:
    """正在累积的语音段。

    IDLE 态下为「未确认候选」（R6：短于 ``min_speech_ms`` 则丢弃）；
    SPEAKING 态下为「活动段」。
    """

    start_ms: int
    last_speech_ms: int
    end_ms: int = 0
    speech_frames: int = 0
    total_frames: int = 0
    peak_prob: float = 0.0
    peak_level_db: float = -120.0
    level_db_sum: float = 0.0
    speech_run: int = 0
    confirmed: bool = False            # 是否曾满足 confirm_frames 连续起口
    max_internal_gap_ms: int = 0
    has_gap: bool = False
    frames: List[Tuple[int, int, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def duration_ms(self) -> int:
        """有效语音跨度：首个语音帧起点 → 最近语音帧终点。"""
        return self.last_speech_ms - self.start_ms

    @property
    def confidence(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return self.speech_frames / self.total_frames

    @property
    def mean_level_db(self) -> float:
        if self.speech_frames <= 0:
            return -120.0
        return self.level_db_sum / self.speech_frames

    # ------------------------------------------------------------------
    def add_speech(self, frame: AudioFrame, confirm_frames: int) -> None:
        gap = frame.ts_start_ms - self.last_speech_ms if self.speech_frames else 0
        if gap > 0:
            self.max_internal_gap_ms = max(self.max_internal_gap_ms, gap)
        self.speech_frames += 1
        self.total_frames += 1
        self.last_speech_ms = frame.ts_end_ms
        self.end_ms = frame.ts_end_ms
        self.peak_prob = max(self.peak_prob, frame.prob)
        self.peak_level_db = max(self.peak_level_db, frame.level_db)
        self.level_db_sum += frame.level_db
        self.speech_run += 1
        if self.speech_run >= confirm_frames:
            self.confirmed = True
        self.frames.append((frame.ts_start_ms, frame.ts_end_ms, frame.prob, frame.level_db))

    def add_quiet(self, frame: AudioFrame, gap_merge_ms: int) -> None:
        """记录一个非语音帧。

        间隙 < ``gap_merge_ms`` 时保留候选（文档 §7-G6），但会打断
        ``speech_run``，起口确认需要重新连续累积。
        """
        self.total_frames += 1
        self.end_ms = frame.ts_end_ms
        gap = frame.ts_start_ms - self.last_speech_ms
        if gap > self.max_internal_gap_ms:
            self.max_internal_gap_ms = gap
        if gap >= gap_merge_ms:
            self.speech_run = 0
        self.frames.append((frame.ts_start_ms, frame.ts_end_ms, frame.prob, frame.level_db))

    def copy(self) -> "Candidate":
        return Candidate(
            start_ms=self.start_ms,
            last_speech_ms=self.last_speech_ms,
            end_ms=self.end_ms,
            speech_frames=self.speech_frames,
            total_frames=self.total_frames,
            peak_prob=self.peak_prob,
            peak_level_db=self.peak_level_db,
            level_db_sum=self.level_db_sum,
            speech_run=self.speech_run,
            confirmed=self.confirmed,
            max_internal_gap_ms=self.max_internal_gap_ms,
            has_gap=self.has_gap,
            frames=list(self.frames),
        )


@dataclass
class Segment:
    """已闭合、可送 ASR 的整段话。"""

    seg_id: int
    utterance_id: int
    seq_in_utterance: int
    start_ms: int                      # 含 pre_roll 后的真实派发起点
    end_ms: int                        # 含 post_roll
    duration_ms: int
    samples: bytes                     # int16 LE 深拷贝，已脱离环形缓冲
    sample_rate: int = 16000
    channels: int = 1
    close_reason: CloseReason = CloseReason.SILENCE
    status: SampleStatus = SampleStatus.CLEAN
    vad_confidence: float = 0.0
    noise_floor_db: float = -60.0
    partial: bool = False              # True = 提前送出的临时段，final 会覆盖同 seg_id
    decision_ms: int = 0               # 判决完成时刻（用于延迟统计）
    speech_start_ms: int = 0           # 未加 pre_roll 的 VAD 起口点
    last_speech_ms: int = 0            # 未加 post_roll 的 VAD 尾点
    meta: dict = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return not self.partial

    def summary(self) -> str:
        return (
            f"seg#{self.seg_id} utt#{self.utterance_id}:{self.seq_in_utterance} "
            f"[{self.start_ms},{self.end_ms}] {self.duration_ms}ms "
            f"{self.close_reason.value}{'' if self.is_final else '/partial'} "
            f"{self.status.value} conf={self.vad_confidence:.2f}"
        )

    def __repr__(self) -> str:  # pragma: no cover - 只用于日志
        return f"Segment({self.summary()}, {len(self.samples)}B)"
