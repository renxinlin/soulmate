# -*- coding: utf-8 -*-
"""语音派发门控。

对应设计文档 §7-G8（唤醒门控）与 §7-G9（TTS 自听 / 回声）。门控与主状态机
**正交**：无论 ``IDLE`` 还是 ``SPEAKING``，gate 关闭时一律不允许新段进入或派发，
这样就能在保持「两状态」约束的前提下接入唤醒与半双工。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class ISpeechGate(ABC):
    """派发许可判定。"""

    @abstractmethod
    def allow(self) -> Tuple[bool, str]:
        """返回 ``(是否允许, 原因)``。原因用于统计与日志。"""


class StaticGate(ISpeechGate):
    """由外部（唤醒模块 / TTS 播放器）显式设置的门控。

    Parameters
    ----------
    armed             : 是否处于唤醒态；``False`` 时只做 VAD 统计不派发
    gate_mode         : ``half`` = TTS 播放期间完全抑制；``full`` = 已有 AEC，允许 barge-in
    ignore_during_tts : ``half`` 模式下是否在播放期间关闸
    arm_timeout_ms    : 唤醒后无交互的自动 disarm 时长；``0`` 表示不自动 disarm
    """

    def __init__(
        self,
        armed: bool = True,
        gate_mode: str = "half",
        ignore_during_tts: bool = True,
        arm_timeout_ms: int = 0,
    ) -> None:
        if gate_mode not in ("half", "full"):
            raise ValueError(f"gate_mode 只能是 half|full，收到 {gate_mode!r}")
        self.armed = armed
        self.gate_mode = gate_mode
        self.ignore_during_tts = ignore_during_tts
        self.arm_timeout_ms = arm_timeout_ms
        self.tts_playing = False
        self.last_activity_ms = 0

    # ------------------------------------------------------------------
    def touch(self, now_ms: int) -> None:
        """记录一次有效交互（派发了一段语音），用于自动 disarm。"""
        self.last_activity_ms = now_ms

    def arm(self, now_ms: int = 0) -> None:
        self.armed = True
        self.last_activity_ms = now_ms

    def disarm(self) -> None:
        self.armed = False

    def set_tts_playing(self, playing: bool) -> None:
        """TTS 播放器在起播/结束时调用。"""
        self.tts_playing = playing

    # ------------------------------------------------------------------
    def allow(self) -> Tuple[bool, str]:
        if not self.armed:
            return False, "not_armed"
        if self.tts_playing and self.ignore_during_tts and self.gate_mode == "half":
            return False, "tts_playing"
        return True, ""

    def check_timeout(self, now_ms: int) -> bool:
        """超时自动 disarm；返回是否发生 disarm。"""
        if (
            self.armed
            and self.arm_timeout_ms > 0
            and self.last_activity_ms > 0
            and now_ms - self.last_activity_ms > self.arm_timeout_ms
        ):
            self.armed = False
            return True
        return False

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StaticGate(armed={self.armed}, mode={self.gate_mode}, "
            f"tts={self.tts_playing}, timeout={self.arm_timeout_ms})"
        )
