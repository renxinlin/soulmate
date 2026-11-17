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
        samples = utt.trim_to_speech()           # 整句 PCM
        frames = utt.frame_detail(cfg.block_ms)  # 逐帧细节数组
"""
from __future__ import annotations

from .collector import (
    BroadcastDispatcher,
    FrameDetail,
    Hole,
    Utterance,
    UtteranceAssembler,
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
