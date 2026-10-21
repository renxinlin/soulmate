# -*- coding: utf-8 -*-
"""
情感引擎 EmotionEngine
======================
管理机器人的情感生命周期：

* 维护当前 ``EmotionState``（PAD 三维连续状态）
* 响应外部刺激（``stimulate`` / ``shift``）驱动情感变化
* 周期性衰减（``decay``）使情感回归中性
* 钩子系统：``on_change`` / ``on_neutral`` / ``on_emotion_shift`` 回调
* 情感边界：当 PAD 值落入某情感区域时自动切换主导标签

典型用法::

    engine = EmotionEngine(
        on_change=lambda old, new: print(f"{old} -> {new}"),
    )
    engine.stimulate(EmotionType.HAPPY, intensity=0.8)
    engine.decay()
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

from .base import EMOTION_VECTORS, EmotionState, EmotionType

logger = logging.getLogger(__name__)

_VALID_HOOKS = frozenset({"on_change", "on_neutral", "on_emotion_shift"})


class EmotionEngine:
    """
    情感引擎：管理情感状态的完整生命周期。

    Parameters
    ----------
    initial_state : 初始情感状态；为 None 则创建中性状态
    decay_factor  : 每次衰减系数 (0, 1]，越大衰减越快
    hooks         : 钩子字典，支持 on_change / on_neutral / on_emotion_shift
    """

    def __init__(
        self,
        initial_state: Optional[EmotionState] = None,
        decay_factor: float = 0.1,
        hooks: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self._state = initial_state or EmotionState()
        self._decay_factor = decay_factor
        self._hooks: Dict[str, Callable] = {}
        if hooks:
            for name, fn in hooks.items():
                if name in _VALID_HOOKS:
                    self._hooks[name] = fn

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def state(self) -> EmotionState:
        """当前情感状态（只读引用）。"""
        return self._state

    @property
    def emotion(self) -> EmotionType:
        """当前主导情感标签的快捷访问。"""
        return self._state.emotion

    @property
    def decay_factor(self) -> float:
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("decay_factor 必须在 (0, 1] 范围内")
        self._decay_factor = value

    @property
    def is_neutral(self) -> bool:
        return self._state.is_neutral()

    # ------------------------------------------------------------------
    # 钩子
    # ------------------------------------------------------------------
    def set_hook(self, name: str, callback: Callable) -> None:
        """注册或更新钩子回调。"""
        if name not in _VALID_HOOKS:
            raise ValueError(f"未知钩子: {name}，可选: {_VALID_HOOKS}")
        self._hooks[name] = callback

    def remove_hook(self, name: str) -> None:
        self._hooks.pop(name, None)

    def _fire_hook(self, name: str, *args: Any) -> None:
        hook = self._hooks.get(name)
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:
            logger.exception("情感钩子 %s 执行失败", name)

    # ------------------------------------------------------------------
    # 情感驱动
    # ------------------------------------------------------------------
    def stimulate(
        self,
        emotion: EmotionType,
        intensity: float = 0.5,
    ) -> EmotionState:
        """
        向指定情感方向偏移。

        Parameters
        ----------
        emotion   : 目标情感类型
        intensity : 偏移强度 [0, 1]，0 无效果，1 直接跳到标准向量

        Returns
        -------
        偏移后的新情感状态副本
        """
        intensity = max(0.0, min(1.0, float(intensity)))
        vec = EMOTION_VECTORS[emotion]
        old = self._state.copy()
        self._state.shift(
            dv=vec[0] * intensity,
            da=vec[1] * intensity,
            dd=vec[2] * intensity,
        )
        self._after_change(old)
        return self._state.copy()

    def shift(
        self,
        dv: float = 0.0,
        da: float = 0.0,
        dd: float = 0.0,
    ) -> EmotionState:
        """直接对 PAD 三轴施加偏移量。"""
        old = self._state.copy()
        self._state.shift(dv=dv, da=da, dd=dd)
        self._after_change(old)
        return self._state.copy()

    def decay(self) -> EmotionState:
        """执行一次情感衰减（向中性回归），由调度器周期性调用。"""
        old = self._state.copy()
        self._state.decay(self._decay_factor)
        self._after_change(old)
        if self._state.is_neutral() and not old.is_neutral():
            self._fire_hook("on_neutral", self._state.copy())
            logger.debug("情感已回归中性")
        return self._state.copy()

    def set_state(
        self,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
    ) -> EmotionState:
        """直接设定情感维度值（None 表示不变）。"""
        old = self._state.copy()
        self._state.update(valence=valence, arousal=arousal, dominance=dominance)
        self._after_change(old)
        return self._state.copy()

    def set_emotion(self, emotion: EmotionType, intensity: float = 1.0) -> EmotionState:
        """直接设定为某情感的标准状态。"""
        intensity = max(0.0, min(1.0, float(intensity)))
        vec = EMOTION_VECTORS[emotion]
        old = self._state.copy()
        self._state.update(
            valence=vec[0] * intensity,
            arousal=vec[1] * intensity,
            dominance=vec[2] * intensity,
        )
        self._after_change(old)
        return self._state.copy()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _after_change(self, old: EmotionState) -> None:
        """变化后统一处理：触发钩子。"""
        if old.emotion != self._state.emotion:
            self._fire_hook("on_emotion_shift", old.emotion, self._state.emotion)
        self._fire_hook("on_change", old, self._state.copy())

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self._state.to_dict(),
            "decay_factor": self._decay_factor,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        hooks: Optional[Dict[str, Callable]] = None,
    ) -> EmotionEngine:
        engine = cls(
            initial_state=EmotionState.from_dict(data["state"]),
            decay_factor=data.get("decay_factor", 0.1),
            hooks=hooks,
        )
        return engine

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"EmotionEngine({self._state!r})"
