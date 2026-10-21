# -*- coding: utf-8 -*-
"""
情感基础类
==========
定义情感模型的核心抽象：

* ``EmotionType``      —— 13 种基本情感枚举
* ``EMOTION_VECTORS``  —— 每种情感在 PAD 空间中的标准向量 (pleasure, arousal, dominance)
* ``EmotionState``     —— 连续情感状态（三维 PAD 模型 + 主导情感标签）

PAD 模型参考
-----------
Mehrabian & Russell (1974) 的 Pleasure-Arousal-Dominance 三维情感空间：

* **Pleasure (愉悦度)**: -1.0 (极度痛苦) → +1.0 (极度愉悦)
* **Arousal  (唤醒度)**: -1.0 (极度困倦) → +1.0 (极度兴奋)
* **Dominance (支配度)**: -1.0 (极度无力) → +1.0 (极度掌控)

每种离散情感对应 PAD 空间中的一个区域，连续维度可以表达更细腻的情感混合。
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Any, Dict, Optional, Tuple


class EmotionType(enum.Enum):
    """基本情感类型枚举。"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    TENDER = "tender"
    RELAXED = "relaxed"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    EXCITED = "excited"
    BORED = "bored"


EMOTION_VECTORS: Dict[EmotionType, Tuple[float, float, float]] = {
    EmotionType.NEUTRAL:   ( 0.0,  0.0,  0.0),
    EmotionType.HAPPY:     ( 0.8,  0.5,  0.3),
    EmotionType.SAD:       (-0.7, -0.4, -0.3),
    EmotionType.ANGRY:     (-0.5,  0.7,  0.4),
    EmotionType.FEARFUL:   (-0.7,  0.6, -0.5),
    EmotionType.SURPRISED: ( 0.0,  0.7,  0.0),
    EmotionType.DISGUSTED: (-0.6,  0.3,  0.2),
    EmotionType.TENDER:    ( 0.6, -0.2, -0.3),
    EmotionType.RELAXED:   ( 0.5, -0.5, -0.2),
    EmotionType.ANXIOUS:   (-0.4,  0.5, -0.4),
    EmotionType.DEPRESSED: (-0.8, -0.6, -0.5),
    EmotionType.EXCITED:   ( 0.7,  0.8,  0.4),
    EmotionType.BORED:     (-0.3, -0.6, -0.2),
}


class EmotionState:
    """
    连续情感状态（PAD 三维模型）。

    属性
    ----
    valence   : 愉悦度 [-1.0, 1.0]
    arousal   : 唤醒度 [-1.0, 1.0]
    dominance : 支配度 [-1.0, 1.0]
    emotion   : 当前主导情感标签（由 PAD 向量自动推断）

    线程安全：内部使用 RLock 保护读写。
    """

    __slots__ = ("_valence", "_arousal", "_dominance", "_emotion",
                 "_lock", "_timestamp")

    def __init__(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.0,
        emotion: Optional[EmotionType] = None,
    ) -> None:
        self._valence = self._clamp(valence)
        self._arousal = self._clamp(arousal)
        self._dominance = self._clamp(dominance)
        self._emotion = emotion or self._infer_emotion()
        self._lock = threading.RLock()
        self._timestamp: float = time.time()

    # ------------------------------------------------------------------
    # 属性（只读，线程安全）
    # ------------------------------------------------------------------
    @property
    def valence(self) -> float:
        with self._lock:
            return self._valence

    @property
    def arousal(self) -> float:
        with self._lock:
            return self._arousal

    @property
    def dominance(self) -> float:
        with self._lock:
            return self._dominance

    @property
    def emotion(self) -> EmotionType:
        with self._lock:
            return self._emotion

    @property
    def timestamp(self) -> float:
        with self._lock:
            return self._timestamp

    @property
    def pad(self) -> Tuple[float, float, float]:
        """返回当前 PAD 向量副本。"""
        with self._lock:
            return (self._valence, self._arousal, self._dominance)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------
    def update(
        self,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
        infer: bool = True,
    ) -> EmotionState:
        """更新情感维度（None 表示不变），返回 self 以支持链式调用。"""
        with self._lock:
            if valence is not None:
                self._valence = self._clamp(valence)
            if arousal is not None:
                self._arousal = self._clamp(arousal)
            if dominance is not None:
                self._dominance = self._clamp(dominance)
            if infer:
                self._emotion = self._infer_emotion()
            self._timestamp = time.time()
        return self

    def shift(
        self,
        dv: float = 0.0,
        da: float = 0.0,
        dd: float = 0.0,
    ) -> EmotionState:
        """情感偏移：在当前值基础上叠加变化量，自动裁剪到 [-1, 1]。"""
        with self._lock:
            self._valence = self._clamp(self._valence + dv)
            self._arousal = self._clamp(self._arousal + da)
            self._dominance = self._clamp(self._dominance + dd)
            self._emotion = self._infer_emotion()
            self._timestamp = time.time()
        return self

    def decay(self, factor: float = 0.1) -> EmotionState:
        """向中性衰减：每个维度按 factor 比例回归 0。"""
        with self._lock:
            self._valence = self._clamp(self._valence * (1.0 - factor))
            self._arousal = self._clamp(self._arousal * (1.0 - factor))
            self._dominance = self._clamp(self._dominance * (1.0 - factor))
            self._emotion = self._infer_emotion()
            self._timestamp = time.time()
        return self

    def is_neutral(self, threshold: float = 0.1) -> bool:
        """是否处于近似中性状态。"""
        with self._lock:
            return (abs(self._valence) < threshold
                    and abs(self._arousal) < threshold
                    and abs(self._dominance) < threshold)

    def distance_to(self, emotion: EmotionType) -> float:
        """当前状态到指定情感标准向量的欧氏距离。"""
        vec = EMOTION_VECTORS[emotion]
        with self._lock:
            return (
                (self._valence - vec[0]) ** 2
                + (self._arousal - vec[1]) ** 2
                + (self._dominance - vec[2]) ** 2
            ) ** 0.5

    def copy(self) -> EmotionState:
        """创建当前状态的独立副本。"""
        with self._lock:
            return EmotionState(
                self._valence, self._arousal, self._dominance, self._emotion,
            )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _infer_emotion(self) -> EmotionType:
        """根据 PAD 值推断最接近的离散情感标签。"""
        best = EmotionType.NEUTRAL
        best_dist = float("inf")
        for etype, vec in EMOTION_VECTORS.items():
            dist = (
                (self._valence - vec[0]) ** 2
                + (self._arousal - vec[1]) ** 2
                + (self._dominance - vec[2]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best = etype
        return best

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "valence": self._valence,
                "arousal": self._arousal,
                "dominance": self._dominance,
                "emotion": self._emotion.value,
                "timestamp": self._timestamp,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmotionState:
        state = cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.0),
            emotion=EmotionType(data["emotion"]) if data.get("emotion") else None,
        )
        if "timestamp" in data:
            state._timestamp = data["timestamp"]
        return state

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            return (
                f"EmotionState({self._emotion.value}, "
                f"V={self._valence:+.2f} A={self._arousal:+.2f} "
                f"D={self._dominance:+.2f})"
            )
