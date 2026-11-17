# -*- coding: utf-8 -*-
"""实时语音活动分段（说话 / 空闲 双状态）。

设计文档：``docs/segment_pipeline_design.md``

对外主要接口::

    from com.robot.segment import SegmentConfig, SegmentEngine, AudioRingBuffer

    cfg = SegmentConfig()
    ring = AudioRingBuffer(cfg)
    asm = UtteranceAssembler(tag="mic")          # 把分段拼回整句（文档 §5.6.1）
    engine = SegmentEngine(cfg, ring, dispatcher=asm)
    # 采集线程
    ring.write(pcm_bytes, frame_start_ms)
    # VAD 线程（滞后 vad_lag_ms）
    engine.step(frame)
    # ASR 线程
    utt = asm.take(timeout=0.2)
    if utt is not None:
        b = utt.to_bundle()                    # 交付载荷：整句 + 语音段 + 元信息
        if b.usable("asr"):
            run_asr(b.speech.to_numpy())       # float32 ±1.0，已按 VAD 边界裁好
        b.full.duration_ms                     # 整句（含 pre/post_roll）供分离模型用
        print(b.json())                        # 落库/日志（不含波形）
"""
from __future__ import annotations

from .collector import (
    AudioView,
    BroadcastDispatcher,
    FrameDetail,
    Hole,
    Utterance,
    UtteranceAssembler,
    UtteranceBundle,
    UtteranceWriter,
)
from .config import ConfigError, SegmentConfig
from .dispatcher import CallbackDispatcher, ISegmentDispatcher, QueueDispatcher
from .gate import ISpeechGate, StaticGate
from .models import (
    AudioFrame,
    Candidate,
    CloseReason,
    SampleStatus,
    SegState,
    Segment,
)
from .ring import AudioRingBuffer
from .stats import SegmentStats
from .vad import EnergyVad, ISegmentVAD, iter_frames, rms_db

__all__ = [
    "AudioFrame",
    "AudioRingBuffer",
    "AudioView",
    "BroadcastDispatcher",
    "Candidate",
    "CallbackDispatcher",
    "CloseReason",
    "ConfigError",
    "EnergyVad",
    "FrameDetail",
    "Hole",
    "ISegmentDispatcher",
    "ISegmentVAD",
    "ISpeechGate",
    "QueueDispatcher",
    "SampleStatus",
    "SegState",
    "Segment",
    "SegmentConfig",
    "SegmentEngine",
    "SegmentStats",
    "StaticGate",
    "Utterance",
    "UtteranceAssembler",
    "UtteranceBundle",
    "UtteranceWriter",
    "iter_frames",
    "rms_db",
]


def __getattr__(name):
    """延迟导出 SegmentEngine，避免 ``import com.robot.segment`` 时的循环依赖。"""
    if name == "SegmentEngine":
        from .engine import SegmentEngine

        return SegmentEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
