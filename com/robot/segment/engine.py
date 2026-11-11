# -*- coding: utf-8 -*-
"""说话 / 空闲 双状态分段状态机。

严格实现设计文档 ``docs/segment_pipeline_design.md``：

* §2  三游标坐标系 ``D ≤ C ≤ W`` 与不变式 I1/I2；
* §5.2 事件 E0–E7；
* §5.3 转移表 T1–T12（注释中以 ``T#`` 标注）；
* §5.4 事件优先级 ``E5 > E4 > E3 > E6 > E2 > E1``；
* §5.5 5 秒截断的截断点择优与同句归属；
* §5.6 深拷贝后立即推进 ``D``（不等 ASR）。

状态机**不直接调用任何模型**：输入是带 ``prob`` 的 :class:`AudioFrame`
（由 :mod:`com.robot.segment.vad` 的某个后端产出），输出是 :class:`Segment`。
因此可以被单测逐帧驱动，也可以原样回放线上日志。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from .config import SegmentConfig
from .dispatcher import ISegmentDispatcher
from .gate import ISpeechGate, StaticGate
from .models import AudioFrame, Candidate, CloseReason, SampleStatus, SegState, Segment
from .ring import AudioRingBuffer
from .stats import SegmentStats

logger = logging.getLogger(__name__)


class SegmentEngine:
    """实时分段状态机。

    Parameters
    ----------
    cfg        : :class:`SegmentConfig`，构造时强制 ``validate()``
    ring       : :class:`AudioRingBuffer`，段音频的唯一来源
    dispatcher : 段出口（ASR 队列等）
    gate       : 唤醒 / TTS 门控（与主状态正交，默认常开）
    stats      : 指标收集器
    on_segment / on_state_change / on_discard : 观测回调
    """

    def __init__(
        self,
        cfg: SegmentConfig,
        ring: AudioRingBuffer,
        dispatcher: ISegmentDispatcher,
        gate: Optional[ISpeechGate] = None,
        stats: Optional[SegmentStats] = None,
        on_segment: Optional[Callable[[Segment], None]] = None,
        on_state_change: Optional[Callable[[SegState, SegState], None]] = None,
        on_discard: Optional[Callable[[str], None]] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.ring = ring
        self.dispatcher = dispatcher
        self.gate = gate or StaticGate(armed=True, gate_mode=cfg.gate_mode,
                                      ignore_during_tts=cfg.ignore_during_tts)
        self.stats = stats or SegmentStats(window_ms=cfg.stat_window_ms)
        self.on_segment = on_segment
        self.on_state_change = on_state_change
        self.on_discard = on_discard

        self._state = SegState.IDLE                 # R3：启动即空闲
        self._c = 0
        self._d = 0
        self._w = 0
        self._grid0: Optional[int] = None           # VAD 帧栅格原点（首个观测到的帧起点）
        self._last_frame_end_ms = 0
        self._cand: Optional[Candidate] = None      # IDLE 下未确认的候选
        self._act: Optional[Candidate] = None       # SPEAKING 下的活动段
        self._seg_id = 0
        self._utt_id = 0
        self._utt_seq = 0
        self._last_close: Optional[CloseReason] = None
        self._partial_emitted = False
        self._gate_was_open = True
        self.pending_flush = False                  # 文档 §5.1：与主状态正交的投递标志

    # ================================================================== 只读
    @property
    def state(self) -> SegState:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state is SegState.SPEAKING

    @property
    def w_ms(self) -> int:
        return self._w

    @property
    def c_ms(self) -> int:
        """VAD 游标 C —— 已判决到的时刻（R1）。"""
        return self._c

    @property
    def d_ms(self) -> int:
        """消费游标 D —— 最近一次已派发/已丢弃段的终点（文档 §2）。"""
        return self._d

    @property
    def active_segment(self) -> Optional[Candidate]:
        return self._act

    # ================================================================== 主入口
    def feed(self, samples: bytes, start_ms: int, vad) -> None:
        """便捷入口：PCM 帧 → VAD 概率 → :meth:`step`。"""
        end_ms = start_ms + len(samples) // (self.cfg.bytes_per_ms or 1)
        frame = AudioFrame(
            ts_start_ms=start_ms,
            ts_end_ms=end_ms,
            prob=vad.prob(samples),
            level_db=_safe_rms_db(samples),
        )
        self.step(frame)

    def step(self, frame: AudioFrame, write_ms: Optional[int] = None) -> None:
        """按帧驱动状态机（文档 §5.8）。``write_ms`` 为采集侧 W 的显式快照。"""
        cfg = self.cfg
        # ---- 游标 ----
        self._w = max(self._w, self.ring.w_ms, frame.ts_end_ms, write_ms or 0)
        self._c = frame.ts_end_ms
        if self._grid0 is None:
            self._grid0 = frame.ts_start_ms
        if frame.ts_start_ms > self._last_frame_end_ms:
            self._on_timeline_gap(frame.ts_start_ms - self._last_frame_end_ms, frame)
        self._last_frame_end_ms = frame.ts_end_ms

        self.stats.push_frame(frame.ts_end_ms, frame.level_db, frame.prob)
        if cfg.warmup_ms > 0 and frame.ts_end_ms <= cfg.warmup_ms:
            return                                   # 文档 §7-G15

        # ---- E5 缓冲压力：优先级最高，数据安全 ----
        if self._w - self._d >= self.ring.capacity_ms - cfg.hard_margin_ms:
            self._handle_ring_pressure()

        gate_ok, gate_why = self._gate_allow()
        # 双阈值迟滞（§7-G13）：起口必须 ≥prob_start；已经在说话时只有跌到 <prob_end
        # 才算静音。中间带既不另起新段，也不推进静音计时，避免句中单帧跌破提前闭合。
        speech = frame.prob >= (cfg.prob_start if self._state is SegState.IDLE else cfg.prob_end)

        if self._state is SegState.IDLE:
            self._step_idle(frame, speech, gate_ok, gate_why)
        else:
            self._step_speaking(frame, speech, gate_ok, gate_why)

        self._advance_idle_watermark()
        self._assert_invariants()

    def flush(self) -> Optional[Segment]:
        """会话结束 / 采集停止：闭合在途段（``EOS``），避免最后半句丢失。"""
        closed: Optional[Segment] = None
        if self._act is not None:
            closed = self._emit_final(self._act, self._act.last_speech_ms, CloseReason.EOS)
            self._act = None
            self._to_state(SegState.IDLE)
        if self._cand is not None:
            self._discard_cand("eos")
        return closed

    def reset(self) -> None:
        self._state = SegState.IDLE
        self._c = self._d = self._w = self._last_frame_end_ms = 0
        self._grid0 = None
        self._cand = self._act = None
        self._partial_emitted = False
        self._last_close = None
        self._gate_was_open = True
        self.pending_flush = False
        self.stats = SegmentStats(window_ms=self.cfg.stat_window_ms)

    # ================================================================== IDLE
    def _step_idle(self, frame: AudioFrame, speech: bool,
                   gate_ok: bool, gate_why: str) -> None:
        cfg = self.cfg
        if speech:
            if self._cand is None:
                self._cand = Candidate(
                    start_ms=frame.ts_start_ms,
                    last_speech_ms=frame.ts_start_ms,   # 首帧 add_speech 时再更新
                    end_ms=frame.ts_end_ms,
                )
            self._cand.add_speech(frame, cfg.confirm_frames)
        elif self._cand is not None:
            silence = frame.ts_end_ms - self._cand.last_speech_ms
            if silence < cfg.min_silence_ms:
                self._cand.add_quiet(frame, cfg.gap_merge_ms)   # T5 前置：短间隙不结束候选
            else:
                # 候选被静音截断（T3 / T4）
                if self._onset_ready(self._cand) and gate_ok:
                    self._emit_final(self._cand, self._cand.last_speech_ms, CloseReason.SILENCE)
                    self._cand = None
                    return
                self._discard_cand("too_short" if self._cand.duration_ms < cfg.min_speech_ms else gate_why or "low_level")
                return

        if self._cand is None:
            return
        if not gate_ok:
            self._discard_cand(gate_why or "gate_closed")       # T9（IDLE 侧）
            return
        # ---- E2_ONSET：R7-a ∧ R7-b ∧ R7-c ----
        if self._onset_ready(self._cand):
            self._to_speaking()

    def _onset_ready(self, cand: Candidate) -> bool:
        """R7 的三条显式守卫 + 能量门限。"""
        cfg = self.cfg
        if cand is None:
            return False
        r7a = cand.start_ms <= self._c                      # VAD 已完整看过起点
        r7b = cand.start_ms >= self._d                      # 是新段，不是旧段重复触发
        r7c = cand.duration_ms >= cfg.min_speech_ms         # "长声音 ≥ 0.3"
        return bool(r7a and r7b and r7c and cand.confirmed and self._level_ok(cand))

    def _to_speaking(self) -> None:
        assert self._cand is not None
        self._act = self._cand
        self._cand = None
        self._partial_emitted = False
        if self._last_close is CloseReason.MAX_DURATION:
            self._utt_seq += 1                              # §5.5：同一句的后续片段
        else:
            self._utt_id += 1
            self._utt_seq = 0
        self._seg_id += 1
        self.stats.note_onset(self._act.start_ms, self._c)
        self._to_state(SegState.SPEAKING)
        logger.debug("onset → SPEAKING utt#%d:%d [%d,%d]", self._utt_id, self._utt_seq,
                     self._act.start_ms, self._act.last_speech_ms)

    # ================================================================== SPEAKING
    def _step_speaking(self, frame: AudioFrame, speech: bool,
                       gate_ok: bool, gate_why: str) -> None:
        cfg = self.cfg
        act = self._act
        assert act is not None

        if speech:
            act.add_speech(frame, cfg.confirm_frames)
        else:
            act.add_quiet(frame, cfg.gap_merge_ms)

        # ---- E4 时长上限（R5）----
        if self._c - act.start_ms >= cfg.max_utterance_ms:
            cut = self._choose_cut(act)
            self._emit_final(act, cut, CloseReason.MAX_DURATION)
            self._act = None
            self._to_state(SegState.IDLE)                   # R5：截断后回到空闲
            # D 已在 _emit_final 中推进到派发末端；只保留其后的帧，
            # 否则新候选起点 < D 会让 R7-b 永远不成立，人还在说却再也不进 SPEAKING
            rest = [f for f in act.frames if f[0] >= self._d]
            if rest and gate_ok:
                self._cand = self._reseed(rest)
            return

        # ---- E3 静音闭合（R4）----
        silence = self._c - act.last_speech_ms
        if silence >= cfg.min_silence_ms:
            if cfg.emit_partial and not self._partial_emitted:
                self._emit_partial(act)                     # §5.7：先出快速结果
            if not cfg.emit_partial or silence >= cfg.long_silence_ms:
                self._emit_final(act, act.last_speech_ms, CloseReason.SILENCE)
                self._act = None
                self._to_state(SegState.IDLE)
                return

        # ---- E6 门控关闭 ----
        if not gate_ok:
            reason = CloseReason.GATE_CLOSED
            if act.duration_ms >= cfg.min_speech_ms and act.confirmed:
                self._emit_final(act, act.last_speech_ms, reason)
            else:
                self._discard_cand(gate_why or "gate_closed", active=True)
            self._act = None
            self._to_state(SegState.IDLE)

    def _choose_cut(self, act: Candidate) -> int:
        """截断点择优（文档 §5.5）：在上限前的回溯窗里挑最安静的帧边界，别切在元音上。"""
        cfg = self.cfg
        lo = act.start_ms + cfg.max_utterance_ms - cfg.cut_search_ms
        hi = act.start_ms + cfg.max_utterance_ms
        quiet_in_win: List[Tuple[float, int]] = []
        any_in_win: List[int] = []
        for s, _e, prob, level in act.frames:
            if lo <= s < hi:
                any_in_win.append(s)
                if prob < cfg.prob_end:
                    quiet_in_win.append((level, s))
        if quiet_in_win:
            return min(quiet_in_win)[1]
        if any_in_win:
            return min(any_in_win)
        return hi

    def _reseed(self, frames: List[Tuple[int, int, float, float]]) -> Optional[Candidate]:
        """截断后用 ``cut`` 之后的帧重建候选，避免把这段语音丢掉。"""
        cand: Optional[Candidate] = None
        for s, e, prob, level in frames:
            f = AudioFrame(ts_start_ms=s, ts_end_ms=e, prob=prob, level_db=level)
            if cand is None:
                cand = Candidate(start_ms=s, last_speech_ms=s, end_ms=e)
            if prob >= self.cfg.prob_start:
                cand.add_speech(f, self.cfg.confirm_frames)
            else:
                cand.add_quiet(f, self.cfg.gap_merge_ms)
        return cand

    # ================================================================== 派发
    def _emit_final(self, cand: Candidate, cut_ms: int, reason: CloseReason) -> Optional[Segment]:
        seg = self._build(cand, cand.start_ms, cut_ms, reason, partial=False)
        self._last_close = reason
        self.pending_flush = True
        ok = self.dispatcher.dispatch(seg)
        self.pending_flush = False
        # §5.6：深拷贝已完成 → D 立即前进，绝不等待 ASR 结果
        self._d = max(self._d, seg.end_ms)
        self.stats.note_offset(cand.last_speech_ms, self._c)
        self.stats.note_close(seg)
        if not ok:
            logger.warning("段派发被丢弃（背压）：%s", seg.summary())
        elif self.on_segment:
            self.on_segment(seg)
        if isinstance(self.gate, StaticGate):
            self.gate.touch(self._c)
        logger.debug("dispatch %s", seg.summary())
        return seg

    def _emit_partial(self, cand: Candidate) -> None:
        seg = self._build(cand, cand.start_ms, cand.last_speech_ms, CloseReason.SILENCE, partial=True)
        self._partial_emitted = True
        ok = self.dispatcher.dispatch(seg)
        # partial 不推进 D：final 会以更宽的区间重读并覆盖同一 seg_id
        self.stats.note_close(seg)
        if ok and self.on_segment:
            self.on_segment(seg)
        logger.debug("dispatch(partial) %s", seg.summary())

    def _advance_idle_watermark(self) -> None:
        """IDLE 且无候选时 ``D`` 跟随 ``C``（文档 §2 I2 的落地细节）。

        纯静音没有任何待派发音频，``D`` 若原地不动，``W - D`` 会一直涨，
        几秒后就会误触 E5 缓冲压力、把「没说的话」当溢出音频告警。
        有候选时不能越过候选起点，否则 pre_roll 想回看的音频会被判为已消费。
        """
        if self._state is SegState.IDLE and self._cand is None and self._act is None:
            self._d = max(self._d, self._c)

    def _align(self, ms: int) -> int:
        """把派发起止对齐到 VAD 帧栅格（向下取整），避免把一个判决帧的音频劈开。"""
        g = self.cfg.block_ms
        if g <= 0 or self._grid0 is None:
            return ms
        return self._grid0 + ((ms - self._grid0) // g) * g

    def _build(self, cand: Candidate, start_ms: int, cut_ms: int,
               reason: CloseReason, partial: bool) -> Segment:
        cfg = self.cfg
        lo = start_ms - cfg.pre_roll_ms               # §7-G7：起口前回溯，防首字缺声
        if not partial and self._utt_seq > 0:
            lo = max(lo, self._d)                     # §5.5：同句续段不与上一段重叠
        s0 = max(0, self._align(lo))
        hi = min(cut_ms + cfg.post_roll_ms, self._c)  # §7-G7：尾点后延，且不越过 C
        e0 = max(self._align(hi), min(cut_ms, self._c))   # 对齐不能把语音尾音切掉
        samples, status = self.ring.read(s0, e0)
        if status is SampleStatus.CLEAN and cand.has_gap:
            status = SampleStatus.GAP
        return Segment(
            seg_id=self._seg_id,
            utterance_id=self._utt_id,
            seq_in_utterance=self._utt_seq,
            start_ms=s0,
            end_ms=e0,
            duration_ms=max(0, e0 - s0),
            samples=samples,
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            close_reason=reason,
            status=status,
            vad_confidence=round(cand.confidence, 3),
            noise_floor_db=round(self.stats.noise_floor_db(-120.0), 1),
            partial=partial,
            decision_ms=self._c,
            speech_start_ms=cand.start_ms,
            last_speech_ms=cand.last_speech_ms,
            meta={
                "peak_level_db": round(cand.peak_level_db, 1),
                "mean_level_db": round(cand.mean_level_db, 1),
                "peak_prob": round(cand.peak_prob, 3),
                "max_internal_gap_ms": cand.max_internal_gap_ms,
                "speech_frames": cand.speech_frames,
                "needs_stitch": reason is CloseReason.MAX_DURATION,
                "also_silence": bool(
                    reason is CloseReason.MAX_DURATION
                    and self._c - cand.last_speech_ms >= cfg.min_silence_ms
                ),
            },
        )

    def _discard_cand(self, why: str, active: bool = False) -> None:
        """T3：丢弃候选但推进 ``D``，否则同一段语音会被反复重新判决（文档 §2 R7-b）。"""
        cand = self._act if active else self._cand
        if cand is not None:
            self._d = max(self._d, cand.last_speech_ms)
        if active:
            self._act = None
        else:
            self._cand = None
        self.stats.note_discard_short()
        if self.on_discard:
            self.on_discard(why)
        logger.debug("discard candidate (%s)", why)

    # ================================================================== 边界
    def _handle_ring_pressure(self) -> None:
        """E5 / I2：缓冲快被写指针追上时强制闭合或丢弃。"""
        if self._act is not None:
            logger.warning("环形缓冲压力，强制闭合 W=%d D=%d", self._w, self._d)
            self._emit_final(self._act, self._act.last_speech_ms, CloseReason.RING_PRESSURE)
            self._act = None
            self._to_state(SegState.IDLE)
            return
        if self._cand is not None:
            self._discard_cand("ring_pressure")
        # 仍无余量：只能跳过未消费区间（会丢音频，属于参数配置错误，必须告警）
        if self._w - self._d > self.ring.capacity_ms:
            lost = self._w - self._d - self.ring.capacity_ms
            logger.error("环形缓冲溢出，丢失 %dms 未派发音频（请加大 ring_capacity_ms）", lost)
            self._d = max(self._d, self._w - self.ring.capacity_ms)

    def _on_timeline_gap(self, gap_ms: int, frame: AudioFrame) -> None:
        """E7：采集丢帧 / 设备重启造成时间轴空洞（文档 §7-G12）。"""
        self.stats.note_gap(gap_ms)
        cand = self._act or self._cand
        if cand is not None:
            cand.has_gap = True
        if gap_ms >= self.cfg.gap_force_close_ms:
            if self._act is not None:
                logger.warning("采集空洞 %dms，强制闭合当前段", gap_ms)
                self._emit_final(self._act, self._act.last_speech_ms, CloseReason.EOS)
                self._act = None
                self._to_state(SegState.IDLE)
            if self._cand is not None:
                self._discard_cand("timeline_gap")
            elif frame.ts_start_ms > self._d:
                self._d = frame.ts_start_ms                  # T11：跳过空洞，保持时间轴一致

    def _gate_allow(self) -> Tuple[bool, str]:
        if isinstance(self.gate, StaticGate):
            self.gate.check_timeout(self._c)
        ok, why = self.gate.allow()
        if not ok and self._gate_was_open:
            self.stats.note_gate_blocked()          # 只在开→关的那一帧计一次
        self._gate_was_open = ok
        return ok, why

    def _level_ok(self, cand: Candidate) -> bool:
        """时长之外的第二道门限：能量（文档 §7-G10）。"""
        cfg = self.cfg
        thr = cfg.dispatch_min_level_db
        floor = self.stats.noise_floor_db(-120.0)
        if floor > -100.0:                       # 窗内非语音帧足够多，底噪才可信
            thr = max(thr, floor + cfg.min_level_over_noise_db)
        return cand.peak_level_db >= thr

    def _to_state(self, new: SegState) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        self.stats.note_state_flip(old, new)
        if self.on_state_change:
            self.on_state_change(old, new)
        logger.debug("state %s -> %s", old.value, new.value)

    def _assert_invariants(self) -> None:
        """I1/I2 运行期断言（文档 §2）。"""
        assert self._d <= self._c, f"I1 违例: D={self._d} > C={self._c}"
        assert self._c <= self._w, f"I1 违例: C={self._c} > W={self._w}"
        assert self._w - self._d <= self.ring.capacity_ms, (
            f"I2 违例: W-D={self._w - self._d} > capacity={self.ring.capacity_ms}"
        )


def _safe_rms_db(samples: bytes) -> float:
    from .vad import rms_db

    try:
        return rms_db(samples)
    except Exception:  # pragma: no cover - 不让电平计算打断判决
        return -120.0
