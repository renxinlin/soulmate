# -*- coding: utf-8 -*-
"""
情感模块功能验证 Demo
=====================
演示 EmotionEngine 的完整使用流程：
  1. 创建引擎 + 注册钩子
  2. 情感刺激（stimulate）
  3. 情感衰减（decay）
  4. 与 Scheduler 集成（EmotionTask 周期衰减）
  5. 序列化 / 反序列化

运行::

    python -m com.robot.emotion.demo
"""
from __future__ import annotations

import time

from .base import EmotionState, EmotionType
from .engine import EmotionEngine
from .emotion_task import EmotionTask


def _on_change(old: EmotionState, new: EmotionState) -> None:
    print(f"  [变化] {old} -> {new}")


def _on_neutral(state: EmotionState) -> None:
    print(f"  [中性] 情感已回归中性: {state}")


def _on_emotion_shift(old_type: EmotionType, new_type: EmotionType) -> None:
    print(f"  [切换] 主导情感: {old_type.value} -> {new_type.value}")


def demo_basic() -> None:
    """基础情感操作演示。"""
    print("=" * 60)
    print("  情感模块 Demo - 基础操作")
    print("=" * 60)

    engine = EmotionEngine(
        hooks={
            "on_change": _on_change,
            "on_neutral": _on_neutral,
            "on_emotion_shift": _on_emotion_shift,
        },
    )
    print(f"\n初始状态: {engine.state}")

    print("\n--- 刺激: 开心 (intensity=0.8) ---")
    engine.stimulate(EmotionType.HAPPY, intensity=0.8)
    print(f"当前状态: {engine.state}")

    print("\n--- 刺激: 兴奋 (intensity=0.3) ---")
    engine.stimulate(EmotionType.EXCITED, intensity=0.3)
    print(f"当前状态: {engine.state}")

    print("\n--- 直接设定: 悲伤 ---")
    engine.set_emotion(EmotionType.SAD, intensity=0.7)
    print(f"当前状态: {engine.state}")

    print("\n--- 衰减 (factor=0.1) ---")
    for i in range(5):
        engine.decay()
        print(f"  第{i + 1}次衰减: {engine.state}")

    print("\n--- 快速衰减到中性 ---")
    while not engine.is_neutral:
        engine.decay()
    print(f"最终状态: {engine.state}")


def demo_scheduler() -> None:
    """调度器集成演示。"""
    print("\n" + "=" * 60)
    print("  情感模块 Demo - 调度器集成")
    print("=" * 60)

    try:
        from ..scheduler import Scheduler
    except ImportError:
        print("  [跳过] 调度器模块未就绪")
        return

    engine = EmotionEngine(
        decay_factor=0.15,
        hooks={
            "on_change": _on_change,
            "on_neutral": _on_neutral,
            "on_emotion_shift": _on_emotion_shift,
        },
    )

    print(f"\n初始状态: {engine.state}")
    print("设定情感: 兴奋 (intensity=0.9)")
    engine.set_emotion(EmotionType.EXCITED, intensity=0.9)
    print(f"当前状态: {engine.state}")

    task = EmotionTask(engine=engine, interval=1)

    sched = Scheduler(workers=2, max_concurrency=1)
    sched.start()
    sched.submit(task)

    print("\n调度器运行中，情感每 1 秒衰减一次...")
    for i in range(8):
        time.sleep(1)
        print(f"  [{i + 1}s] {engine.state}")
        if engine.is_neutral:
            print("  已到达中性状态，停止演示")
            break

    task.cancel()
    sched.stop()
    print(f"\n最终状态: {engine.state}")


def demo_serialization() -> None:
    """序列化演示。"""
    print("\n" + "=" * 60)
    print("  情感模块 Demo - 序列化")
    print("=" * 60)

    engine = EmotionEngine(decay_factor=0.12)
    engine.stimulate(EmotionType.TENDER, intensity=0.6)
    print(f"\n原始状态: {engine.state}")

    data = engine.to_dict()
    print(f"序列化:   {data}")

    restored = EmotionEngine.from_dict(data)
    print(f"还原状态: {restored.state}")
    print(f"衰减系数: {restored.decay_factor}")


if __name__ == "__main__":
    demo_basic()
    demo_scheduler()
    demo_serialization()
    print("\n" + "=" * 60)
    print("  Demo 完成!")
    print("=" * 60)
