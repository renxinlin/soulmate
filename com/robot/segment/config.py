# -*- coding: utf-8 -*-
"""分段模块配置。

对应设计文档 §4。**容量约束不等式在 :meth:`SegmentConfig.validate` 中强制校验**，
调参破坏约束时启动即失败，而不是运行到某一句随机丢字（文档 §7-G5）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class ConfigError(ValueError):
    """参数组合破坏设计约束。"""


@dataclass
class SegmentConfig:
    # ---- 采集 ----
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 32                 # 采集块 = VAD 帧（512 样本 @16k）
    ring_capacity_ms: int = 8192       # 1<<17 样本 @16k；6000ms 在默认参数下不成立

    # ---- 游标（文档 §2）----
    vad_lag_ms: int = 500              # R1：C = W - vad_lag
    hard_margin_ms: int = 500          # I2 保护带

    # ---- 判决窗（R2，读法 A：统计窗，不用于触发；窗按帧连续滑动，步长即 block_ms）----
    stat_window_ms: int = 2000

    # ---- 帧级阈值（双阈值迟滞，文档 §7-G13）----
    prob_start: float = 0.55
    prob_end: float = 0.35
    confirm_frames: int = 6            # 连续 6 帧 ≈ 192ms

    # ---- 段长度约束 ----
    min_speech_ms: int = 300           # R6 / R7-c
    min_silence_ms: int = 300          # R4
    long_silence_ms: int = 650         # 句末二次确认（仅 emit_partial 模式使用）
    gap_merge_ms: int = 200            # 文档 §7-G6
    gap_force_close_ms: int = 200      # 采集空洞 ≥ 此值 → 强制闭合（文档 §7-G12）
    max_utterance_ms: int = 5000       # R5
    cut_search_ms: int = 500           # 截断点择优回溯窗（文档 §5.5）
    pre_roll_ms: int = 200             # 文档 §7-G7
    post_roll_ms: int = 100

    # ---- 能量门限（文档 §7-G10）----
    dispatch_min_level_db: float = -45.0
    min_level_over_noise_db: float = 6.0

    # ---- 派发 ----
    emit_partial: bool = False         # False = batch（R7 原口径），True = 流式提前送
    # 投递队列容量不属于状态机参数：由 QueueDispatcher(maxsize=…) 在装配时指定（文档 §5.6）

    # ---- 启动 / 边界 ----
    warmup_ms: int = 500               # 文档 §7-G15：丢弃上电 pop 音

    # ---- 门控默认值（文档 §7-G8 / §7-G9）----
    gate_mode: str = "half"            # half | full（full 需 AEC）
    ignore_during_tts: bool = True

    def validate(self) -> None:
        """校验参数组合，违反约束抛 :class:`ConfigError`。"""
        c = self
        if c.sample_rate <= 0 or c.block_ms <= 0 or c.channels != 1:
            raise ConfigError(f"采集参数非法: rate={c.sample_rate} block={c.block_ms} ch={c.channels}")
        if c.prob_end >= c.prob_start:
            raise ConfigError(f"prob_end({c.prob_end}) 必须小于 prob_start({c.prob_start})，否则无迟滞")
        if c.long_silence_ms < c.min_silence_ms:
            raise ConfigError("long_silence_ms 必须 >= min_silence_ms")
        if c.gap_merge_ms > c.min_silence_ms:
            raise ConfigError("gap_merge_ms 必须 <= min_silence_ms，否则间隙合并会吞掉句末判定")

        need = (
            c.max_utterance_ms
            + c.vad_lag_ms
            + c.pre_roll_ms
            + c.post_roll_ms
            + c.hard_margin_ms
        )
        if c.ring_capacity_ms < need:
            raise ConfigError(
                "环形缓冲不足：ring_capacity_ms={cap} < max_utterance({mut}) + vad_lag({lag})"
                " + pre_roll({pre}) + post_roll({post}) + hard_margin({mar}) = {need}"
                "（文档 §7-G5；解法：加大 ring 或降低 max_utterance_ms）".format(
                    cap=c.ring_capacity_ms,
                    mut=c.max_utterance_ms,
                    lag=c.vad_lag_ms,
                    pre=c.pre_roll_ms,
                    post=c.post_roll_ms,
                    mar=c.hard_margin_ms,
                    need=need,
                )
            )
        if c.max_utterance_ms <= c.cut_search_ms:
            raise ConfigError("max_utterance_ms 必须大于 cut_search_ms")
        if c.min_speech_ms < c.confirm_frames * c.block_ms:
            raise ConfigError(
                f"min_speech_ms({c.min_speech_ms}) 小于 confirm_frames 覆盖时长"
                f"({c.confirm_frames * c.block_ms})，起口确认将永不生效"
            )

    # ------------------------------------------------------------------
    @property
    def bytes_per_ms(self) -> int:
        return self.sample_rate * 2 * self.channels // 1000

    @property
    def block_samples(self) -> int:
        return self.sample_rate * self.block_ms // 1000

    @property
    def block_bytes(self) -> int:
        return self.block_samples * 2 * self.channels

    def latency_budget(self) -> Tuple[int, int, int]:
        """返回 (起口延迟, 闭合延迟, 硬截断延迟) 毫秒上界，单位 ms。

        `min_speech/min_silence` 是按帧数完的，所以真实延迟要向上取整到 ``block_ms``
        的栅格（300ms → 10 帧 = 320ms）；``confirm_frames×block`` 不额外串联，
        它落在 min_speech 窗内（否则 :meth:`validate` 直接拒绝）。文档 §6。
        """
        c = self

        def on_grid(ms: int) -> int:
            return -(-ms // c.block_ms) * c.block_ms if c.block_ms > 0 else ms

        onset = c.vad_lag_ms + on_grid(c.min_speech_ms)
        silence = c.long_silence_ms if c.emit_partial else c.min_silence_ms
        return onset, c.vad_lag_ms + on_grid(silence), c.vad_lag_ms + c.max_utterance_ms
