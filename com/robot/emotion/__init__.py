# -*- coding: utf-8 -*-
"""
com.robot.emotion —— 机器人情感引擎
==================================
基于 PAD（Pleasure-Arousal-Dominance）三维连续情感模型，提供：

* ``EmotionType``      —— 13 种基本情感枚举
* ``EMOTION_VECTORS``  —— 每种情感在 PAD 空间中的标准向量
* ``EmotionState``     —— 连续情感状态（三维 PAD + 自动情感标签推断）
* ``EmotionEngine``    —— 情感引擎（刺激、衰减、钩子、序列化）
* ``EmotionTask``      —— 与调度框架集成的周期衰减任务

快速开始::

    from com.robot.emotion import EmotionEngine, EmotionType

    engine = EmotionEngine()
    engine.stimulate(EmotionType.HAPPY, intensity=0.8)
    print(engine.state)   # EmotionState(happy, V=+0.64 A=+0.40 D=+0.24)
    engine.decay()
"""
from .base import EMOTION_VECTORS, EmotionState, EmotionType
from .emotion_task import EmotionTask
from .engine import EmotionEngine

__all__ = [
    "EmotionType",
    "EMOTION_VECTORS",
    "EmotionState",
    "EmotionEngine",
    "EmotionTask",
]
