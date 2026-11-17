# -*- coding: utf-8 -*-
"""``com.robot.segment`` 的 golden 测试（设计文档 §8 的 TC1–TC12 + 若干补充）。

运行方式（仓库根目录）::

    python -m unittest com.robot.segment.test_segment -v
    python com/robot/segment/test_segment.py            # 亦可独立运行

测试直接构造带 ``prob`` 的 :class:`AudioFrame` 逐帧驱动状态机，
不依赖 numpy / funasr / sherpa-onnx，因此可在任何环境（包括 CI 与 RK 板子）跑。

写入环形缓冲的 PCM 采用「每块首样本 = 该块起始毫秒」的编码，
于是 :class:`Segment.samples` 的首样本可以直接还原出派发起点，
用来精确校验 pre-roll 与环形回绕读取是否正确（派发边界对齐到帧栅格，见 §7-G7）。

``TestUtteranceAssembly`` 覆盖的是**派发之后**的整句装配层（``collector.py``，文档 §5.6）：
硬截断拼接、partial 不入波形、空洞补零 / 重叠裁头、逐帧细节数组、队列溢出，
其中一条端到端用例直接把装配器挂在真状态机出口上。
交付载荷（``to_bundle``，文档 §5.6.1）也验一批：整句 / 语音双视图的时间映射与切片、
补零区间随视图裁剪、以及 ``separation`` / ``asr`` / ``voiceprint`` 的可用性判定
——正空洞拦分离、空语音拦 ASR、短语音与任何 suspect 拦声纹。

``TestCaptureDrain`` 覆盖的是**派发之前**的采集侧接缝（``demo2.pump``，文档 §9.2）：
起流期间攒下的积压必须在第一个块之前丢掉，于是状态机处理的是「当前时刻」的语音，
积压期间的丢弃计数也不会混进会话统计。
"""
from __future__ import annotations

import array
import math
import os
import sys
import unittest
import wave

if __package__ in (None, ""):  # 直接以脚本方式运行时补 sys.path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from com.robot.segment.collector import (
    AudioView, BroadcastDispatcher, Utterance, UtteranceAssembler, UtteranceBundle,
    UtteranceWriter)
from com.robot.segment.config import ConfigError, SegmentConfig
from com.robot.segment.dispatcher import CallbackDispatcher, QueueDispatcher
from com.robot.segment.engine import SegmentEngine
from com.robot.segment.gate import StaticGate
from com.robot.segment.models import (
    AudioFrame, CloseReason, SampleStatus, SegState, Segment)
from com.robot.segment.ring import AudioRingBuffer
from com.robot.segment.vad import (
    EnergyVad,
    ScriptedVad,
    WindowSegmentVad,
    parse_segment_result,
    rms_db,
)


def block_pcm(start_ms: int, n_samples: int) -> bytes:
    """首样本编码块起始毫秒，其余为小 amplitude（保证 RMS 高于能量门限）。"""
    a = array.array("h", [0]) * n_samples
    a[0] = min(32767, start_ms)
    for i in range(1, n_samples):
        a[i] = 3000 if i % 2 else -3000
    return a.tobytes()


class Harness:
    """把「写环形缓冲 + 造帧 + 驱动状态机」收在一处。"""

    def __init__(self, cfg: SegmentConfig = None, gate: StaticGate = None, dispatcher=None):
        self.cfg = cfg or SegmentConfig(warmup_ms=0)
        self.cfg.validate()
        self.ring = AudioRingBuffer(self.cfg)
        self.dispatched = []
        self.discards = []
        self.dispatcher = dispatcher or CallbackDispatcher(self.dispatched.append)
        self.gate = gate if gate is not None else StaticGate(armed=True, gate_mode="half")
        self.engine = SegmentEngine(
            self.cfg, self.ring, self.dispatcher, gate=self.gate, on_discard=self.discards.append
        )
        self.t = 0                                     # 下一帧的起始毫秒
        self.n = self.cfg.block_samples
        self.onset_ms = -1                             # 最近一次 speak() 的真实起口时刻
        self.tail_ms = -1                              # 最近一次 quiet() 的真实收尾时刻

    # ---- 时间推进原语 ----
    def frames(self, ms: int, prob: float, level: float = -20.0) -> "Harness":
        for _ in range(max(1, int(round(ms / self.cfg.block_ms)))):
            self.push(prob=prob, level=level)
        return self

    def speak(self, ms: int, prob: float = 0.9, level: float = -20.0) -> "Harness":
        self.onset_ms = self.t
        return self.frames(ms, prob, level=level)

    def quiet(self, ms: int, prob: float = 0.0, level: float = -70.0) -> "Harness":
        self.tail_ms = self.t
        return self.frames(ms, prob, level=level)

    def push(self, prob: float, level: float = -20.0, discontinuous: bool = False,
             skip_ms: int = 0) -> "Harness":
        start = self.t + skip_ms
        end = start + self.cfg.block_ms
        self.ring.write(block_pcm(start, self.n), start)
        self.engine.step(
            AudioFrame(
                ts_start_ms=start,
                ts_end_ms=end,
                prob=prob,
                level_db=level,
                discontinuous=discontinuous or skip_ms > 0,
            )
        )
        self.t = end
        return self

    # ---- 断言辅助 ----
    @property
    def finals(self):
        return [s for s in self.dispatched if s.is_final]

    @property
    def partials(self):
        return [s for s in self.dispatched if s.partial]

    def grid_floor(self, ms: int) -> int:
        """派发边界对齐到帧栅格后的时刻（测试里栅格原点为 0）。"""
        return ms // self.cfg.block_ms * self.cfg.block_ms

    def first_sample_ms(self, seg) -> int:
        a = array.array("h")
        a.frombytes(seg.samples[:2])
        return a[0]


# ======================================================================
class TestStateMachine(unittest.TestCase):
    def test_tc01_pure_silence_stays_idle(self):
        """TC1：纯静音 10s → 全程 IDLE、0 段，且 D 跟随 C（R4 / I2）。"""
        h = Harness()
        h.quiet(10_000)
        self.assertEqual(h.dispatched, [])
        self.assertIs(h.engine.state, SegState.IDLE)
        self.assertEqual(h.engine.d_ms, h.engine.c_ms, "空闲无待发音频时 D 必须跟随 C")
        self.assertGreater(h.engine.d_ms, h.ring.capacity_ms, "10s 静音早已超过缓冲容量")
        self.assertEqual(h.engine.stats.report()["close_counts"]["ring"], 0, "静音不该触发缓冲压力")

    def test_tc02_transient_discarded_and_cursor_advances(self):
        """TC2：0.2s 瞬态 → 保持 IDLE、0 段，但 D 必须前进（R6 / T3）。"""
        h = Harness()
        h.quiet(500).speak(200).quiet(800)
        self.assertEqual(h.dispatched, [])
        self.assertGreater(h.engine.d_ms, h.onset_ms, "丢弃候选必须推进 D，否则同一段会被反复重判")
        self.assertLessEqual(h.engine.d_ms, h.engine.c_ms)
        self.assertEqual(h.engine.stats.discarded_short, 1)
        self.assertIs(h.engine.state, SegState.IDLE)

    def test_tc03_normal_closure_with_roll(self):
        """TC3：1s 语音 + 1s 静音 → 1 段，含 pre/post roll（R4 / R7）。"""
        h = Harness()
        h.quiet(1000).speak(1000).quiet(1000)
        self.assertEqual(len(h.finals), 1)
        seg = h.finals[0]
        self.assertIs(seg.close_reason, CloseReason.SILENCE)
        self.assertEqual(seg.speech_start_ms, h.onset_ms)
        self.assertEqual(seg.start_ms, h.grid_floor(h.onset_ms - h.cfg.pre_roll_ms),
                         "§7-G7：起点必须前扩 pre_roll（向下对齐到帧栅格）")
        self.assertLessEqual(h.cfg.pre_roll_ms, h.onset_ms - seg.start_ms)
        self.assertLess(h.onset_ms - seg.start_ms, h.cfg.pre_roll_ms + h.cfg.block_ms)
        self.assertEqual(seg.end_ms, h.grid_floor(seg.last_speech_ms + h.cfg.post_roll_ms))
        self.assertEqual(h.first_sample_ms(seg), seg.start_ms, "派发的音频必须真的始于 start_ms")
        self.assertEqual(len(seg.samples), (seg.end_ms - seg.start_ms) * h.cfg.bytes_per_ms)
        self.assertIs(h.engine.state, SegState.IDLE)

    def test_tc04_max_duration_cut_keeps_same_utterance(self):
        """TC4：6s 连续语音 → 至少 2 段，同 utterance，段间不重叠不丢音（R5 + §5.5）。"""
        h = Harness()
        h.quiet(200).speak(6000).quiet(800)
        finals = h.finals
        self.assertGreaterEqual(len(finals), 2, "超过 5s 必须截断")
        first = finals[0]
        self.assertIs(first.close_reason, CloseReason.MAX_DURATION)
        self.assertTrue(first.meta["needs_stitch"])
        self.assertEqual(len({s.utterance_id for s in finals[:2]}), 1, "截断片段必须归到同一句")
        self.assertEqual([s.seq_in_utterance for s in finals[:2]], [0, 1])
        # §5.5：续段起点就是上一段终点——既不重叠也不留空洞
        self.assertEqual(finals[1].start_ms, first.end_ms)
        self.assertEqual(h.first_sample_ms(finals[1]), finals[1].start_ms)
        self.assertLessEqual(first.duration_ms, h.cfg.max_utterance_ms + h.cfg.block_ms)
        self.assertGreaterEqual(finals[1].duration_ms, h.cfg.min_speech_ms)

    def test_tc05_short_gap_merged_into_one_segment(self):
        """TC5：语音中间 0.15s 间隙 → 仍是 1 段（§7-G6 gap merge）。"""
        h = Harness()
        h.speak(600).quiet(150).speak(600).quiet(700)
        self.assertEqual(len(h.finals), 1)
        self.assertGreaterEqual(h.finals[0].meta["max_internal_gap_ms"], 150 - h.cfg.block_ms)

    def test_tc06_long_pause_splits_into_two_utterances(self):
        """TC6：语音中间 0.4s 停顿 → 2 段（G1 场景，靠 utterance_id 由下游拼接）。"""
        h = Harness()
        h.speak(600).quiet(400).speak(600).quiet(700)
        self.assertEqual(len(h.finals), 2)
        self.assertNotEqual(h.finals[0].utterance_id, h.finals[1].utterance_id)
        self.assertEqual(h.finals[0].close_reason, h.finals[1].close_reason)

    def test_tc07_onset_latency_and_preroll_coverage(self):
        """TC7：起口后很快判决 → 派发音频仍必须包含起口前 200ms（§7-G7）。"""
        h = Harness()
        h.quiet(1000).speak(600).quiet(600)
        seg = h.finals[0]
        self.assertEqual(h.first_sample_ms(seg), seg.start_ms)
        self.assertGreaterEqual(seg.speech_start_ms - seg.start_ms, h.cfg.pre_roll_ms,
                                "pre_roll 音频必须在派发包络内")
        self.assertLess(seg.speech_start_ms - seg.start_ms, h.cfg.pre_roll_ms + h.cfg.block_ms,
                        "对齐到帧栅格只会多给，不会少给")
        onset = h.engine.stats.onset_latencies[0]
        self.assertGreaterEqual(onset, h.cfg.min_speech_ms)
        self.assertLessEqual(onset, h.cfg.min_speech_ms + 2 * h.cfg.block_ms, "起口延迟 = min_speech")

    def test_tc08_priority_max_duration_over_silence(self):
        """TC8：上限与静音同时成立 → 取 MAX_DURATION 并记 also_silence（§5.4）。"""
        h = Harness()
        h.speak(4700).quiet(400)
        finals = h.finals
        self.assertEqual(len(finals), 1)
        self.assertIs(finals[0].close_reason, CloseReason.MAX_DURATION)
        self.assertTrue(finals[0].meta["also_silence"])

    def test_tc09_tts_self_hearing_suppressed(self):
        """TC9：半双工门控——TTS 播放期间的语音不派发（§7-G9）。"""
        gate = StaticGate(armed=True, gate_mode="half", ignore_during_tts=True)
        gate.set_tts_playing(True)
        h = Harness(gate=gate)
        h.speak(1500).quiet(600)
        self.assertEqual(h.dispatched, [])
        self.assertEqual(h.engine.stats.gate_blocked, 1, "开→关只计一次")
        self.assertIs(h.engine.state, SegState.IDLE)

    def test_tc09b_disarmed_gate_blocks_dispatch(self):
        """TC9b：未唤醒 → 不派发，但 VAD 统计继续（§7-G8）。"""
        gate = StaticGate(armed=False)
        h = Harness(gate=gate)
        h.speak(1500).quiet(600)
        self.assertEqual(h.dispatched, [])
        self.assertGreater(h.engine.stats.total_frames, 0)
        self.assertEqual(h.discards and h.discards[-1], "not_armed")

    def test_tc09c_barge_in_allowed_in_full_mode(self):
        """TC09c：full（有 AEC）模式下 TTS 期间仍可分段。"""
        gate = StaticGate(armed=True, gate_mode="full")
        gate.set_tts_playing(True)
        h = Harness(gate=gate)
        h.speak(1000).quiet(600)
        self.assertEqual(len(h.finals), 1)

    def test_tc10_capture_gap_forces_close(self):
        """TC10：注入 300ms 采集空洞 → 强制闭合且标记 GAP（§7-G12）。"""
        h = Harness()
        h.speak(800)
        h.push(prob=0.9, skip_ms=300)      # discontinuous
        h.speak(100).quiet(600)
        finals = h.finals
        self.assertGreaterEqual(len(finals), 1)
        self.assertIs(finals[0].close_reason, CloseReason.EOS)
        self.assertIs(finals[0].status, SampleStatus.GAP)
        self.assertEqual(h.engine.stats.gap_ms, 300)

    def test_tc11_backpressure_does_not_stall_cursor(self):
        """TC11：消费队列塞满 → D 仍前进、不抛异常（§5.6 / §7-G11）。"""
        q = QueueDispatcher(maxsize=1)
        h = Harness(dispatcher=q)
        h.speak(600).quiet(600).speak(600).quiet(600).speak(600).quiet(600)
        self.assertEqual(h.engine.stats.report()["final_segments"], 3, "状态机自己切了 3 段")
        st = q.stats()
        self.assertGreaterEqual(st["dropped"], 1, "队列满且无人消费时必须丢弃并计数")
        self.assertEqual(len(q.drain()), 1, "队列里只留得下一条")
        self.assertGreater(h.engine.d_ms, 0, "D 不能等待 ASR 消费")
        self.assertIs(h.engine.state, SegState.IDLE)

    def test_tc11b_partial_evicted_final_kept(self):
        """TC11b：背压时优先丢 partial、保 final。"""
        q = QueueDispatcher(maxsize=1)
        cfg = SegmentConfig(warmup_ms=0, emit_partial=True)
        h = Harness(cfg=cfg, dispatcher=q)
        h.speak(800).quiet(400)          # partial 入队，占住唯一那一格
        self.assertEqual(q.qsize(), 1)
        h.speak(600).quiet(800)          # 继续说话 → final 挤掉队列里的 partial
        st = q.stats()
        self.assertEqual(st["dropped"], 1, "被挤掉的必须是 partial")
        self.assertEqual(st["delivered"], 2)
        left = q.drain()
        self.assertEqual(len(left), 1)
        self.assertTrue(left[0].is_final, "final 必须留下")

    def test_tc12_capacity_assert(self):
        """TC12：6000ms 缓冲 + 5000ms 上限 → 启动即失败（§7-G5）。"""
        with self.assertRaises(ConfigError) as ctx:
            SegmentConfig(ring_capacity_ms=6000, warmup_ms=0).validate()
        self.assertIn("环形缓冲不足", str(ctx.exception))
        # 满足约束的组合必须通过
        SegmentConfig(ring_capacity_ms=8192, warmup_ms=0).validate()

    def test_tc13_emit_partial_then_final(self):
        """TC13：emit_partial 模式下先出 partial 再出 final，同一 seg_id 覆盖。"""
        cfg = SegmentConfig(warmup_ms=0, emit_partial=True)
        h = Harness(cfg=cfg)
        h.speak(800).quiet(400)
        self.assertEqual(len(h.partials), 1)
        self.assertEqual(h.finals, [])
        self.assertIs(h.engine.state, SegState.SPEAKING, "partial 之后不能提前离开 SPEAKING")
        h.speak(600).quiet(800)          # 停顿后又接着说 → final 区间必然更宽
        self.assertEqual(len(h.finals), 1)
        p, f = h.partials[0], h.finals[0]
        self.assertEqual(p.seg_id, f.seg_id, "同 seg_id → 下游按覆盖处理，不重复计费")
        self.assertLess(p.end_ms, f.end_ms, "final 必须覆盖 partial 的区间")
        self.assertGreaterEqual(h.engine.d_ms, f.end_ms)

    def test_tc14_hysteresis_and_low_level_rejection(self):
        """TC14：低电平长语音（远场空调）→ 时长够但能量门限拦下（§7-G10）。"""
        h = Harness()
        h.speak(1200, prob=0.9)          # 正常电平
        h.quiet(400)
        self.assertEqual(len(h.finals), 1)
        h2 = Harness(cfg=SegmentConfig(warmup_ms=0, dispatch_min_level_db=-30.0))
        h2.frames(1200, prob=0.9, level=-35.0)
        h2.quiet(400)
        self.assertEqual(h2.finals, [], "低于能量门限不应派发")
        self.assertEqual(len(h2.discards), 1)

    def test_tc14b_noise_floor_ignores_speech(self):
        """TC14b：底噪只能由非语音帧估计，否则连续说话会被自己的电平拦死。"""
        h = Harness()
        h.quiet(2500).speak(3000)        # 窗内前段是 -70dB 底噪
        h.quiet(600)
        self.assertEqual(len(h.finals), 1, "底噪估计混入语音帧时这里会变成 0 段")
        self.assertLess(h.engine.stats.noise_floor_db(), -60.0)

    def test_tc21_release_threshold_hysteresis(self):
        """TC21：SPEAKING 中跌进 [prob_end, prob_start) 中间带不算静音（§7-G13）。"""
        h = Harness()
        h.speak(600)                              # ≥prob_start → SPEAKING
        h.frames(500, prob=0.45)                  # 中间带：既不断段也不推进静音计时
        h.speak(200, prob=0.9)
        h.quiet(400)
        self.assertEqual(len(h.finals), 1, "中间带不能被当成 300ms 静音提前闭合")
        h2 = Harness()
        h2.speak(600).frames(500, prob=0.20).quiet(200)   # 真静音（<prob_end）
        self.assertEqual(len(h2.finals), 1, "跌破 prob_end 必须照常计时")
        h3 = Harness()
        h3.frames(2000, prob=0.45).quiet(400)     # IDLE 侧：中间带不足以起口
        self.assertEqual(h3.dispatched, [], "未达 prob_start 不得进入 SPEAKING")

    def test_tc15_ring_wrap_read(self):
        """TC15：环形回绕后读取的音频仍要正确（时间轴 → 下标映射）。"""
        cfg = SegmentConfig(warmup_ms=0, ring_capacity_ms=2048, max_utterance_ms=800,
                            vad_lag_ms=300, hard_margin_ms=200)
        h = Harness(cfg=cfg)
        for _ in range(4):
            h.speak(900).quiet(400)
        self.assertGreaterEqual(len(h.finals), 3)
        self.assertGreater(h.engine.w_ms, h.ring.capacity_ms, "必须真的发生过回绕")
        for seg in h.finals:
            self.assertIs(seg.status, SampleStatus.CLEAN, seg.summary())
            self.assertEqual(h.first_sample_ms(seg), seg.start_ms, "回绕读取错位：%s" % seg.summary())
            self.assertGreater(len(seg.samples), 0)

    def test_tc16_invariants_hold_throughout(self):
        """TC16：全程满足 D ≤ C ≤ W 与 I2（step 内部亦 assert）。"""
        h = Harness()
        h.quiet(300).speak(1200).quiet(500).speak(5200).quiet(900).speak(700).quiet(900)
        self.assertLessEqual(h.engine.d_ms, h.engine.c_ms)
        self.assertLessEqual(h.engine.c_ms, h.engine.w_ms)
        self.assertLessEqual(h.engine.w_ms - h.engine.d_ms, h.ring.capacity_ms)
        rep = h.engine.stats.report()
        self.assertGreater(rep["final_segments"], 0)
        self.assertEqual(rep["close_counts"][CloseReason.RING_PRESSURE.value], 0)

    def test_tc17_flush_closes_tail(self):
        """TC17：会话结束时在途段必须以 EOS 闭合，否则最后半句丢失（§7-G15）。"""
        h = Harness()
        h.speak(800)                      # 说话中直接停采集
        seg = h.engine.flush()
        self.assertIsNotNone(seg)
        self.assertIs(seg.close_reason, CloseReason.EOS)
        self.assertIs(h.engine.state, SegState.IDLE)


# ======================================================================
class TestVadBackends(unittest.TestCase):
    def test_parse_segment_result_ms_unit(self):
        res = [{"value": [[120, 940], [1500, 2000]]}, {"value": []}, {}, None]
        self.assertEqual(parse_segment_result(res), [(120, 940), (1500, 2000)])
        self.assertEqual(parse_segment_result([]), [])
        # 单位是毫秒：400ms 的段不应被当成 0.4 个采样点
        self.assertGreaterEqual(parse_segment_result(res)[0][1] - parse_segment_result(res)[0][0], 400)

    def test_energy_vad_separates_tone_from_silence(self):
        vad = EnergyVad(threshold_db=-45.0)
        tone = array.array("h", [8000 if i % 2 else -8000 for i in range(512)]).tobytes()
        self.assertGreater(vad.prob(tone), 0.5)
        silence = b"\x00" * 1024
        p = 1.0
        for _ in range(40):
            p = vad.prob(silence)
        self.assertLess(p, 0.35, "释放拖尾必须最终衰减到静音")
        self.assertLess(rms_db(b"\x00" * 1024), -100)

    def test_scripted_vad_sequence(self):
        vad = ScriptedVad([0.9, 0.9, 0.1], default=0.0)
        self.assertEqual([vad.prob(b"") for _ in range(4)], [0.9, 0.9, 0.1, 0.0])

    def test_feed_end_to_end_with_energy_vad(self):
        """TC18：走 feed()（PCM → VAD → 状态机）的完整链路。"""
        cfg = SegmentConfig(warmup_ms=0)
        h = Harness(cfg=cfg)
        vad = EnergyVad(threshold_db=-50.0)
        tone = block_pcm(0, cfg.block_samples)
        silence = b"\x00" * cfg.block_bytes
        t = 0

        def push(data):
            nonlocal t
            h.ring.write(data, t)
            h.engine.feed(data, t, vad)
            t += cfg.block_ms

        for _ in range(20):
            push(silence)
        for _ in range(50):
            push(tone)
        for _ in range(40):
            push(silence)
        self.assertEqual(len(h.finals), 1, "能量 VAD + 状态机应切出 1 段")
        seg = h.finals[0]
        self.assertIs(seg.close_reason, CloseReason.SILENCE)
        self.assertGreaterEqual(seg.meta["speech_frames"], 50, "语音帧数不得少于实际 50 块")
        self.assertLessEqual(seg.speech_start_ms, 20 * cfg.block_ms, "起口点不得晚于实际起口")
        self.assertGreaterEqual(seg.last_speech_ms, 70 * cfg.block_ms, "尾点不得早于实际收尾")

    def test_window_segment_vad_aligns_prob_stream(self):
        """TC19：段列表型 VAD（FSMN）适配后，概率流与帧一一对应，滞后恒定为 lag_ms。"""
        cfg = SegmentConfig(warmup_ms=0)
        block = cfg.block_ms
        window_ms, stride_ms = 1920, 320                 # 必须落在帧栅格上
        win_blocks = window_ms // block                  # 60
        stride_blocks = stride_ms // block               # 10
        warm = win_blocks - 1                            # 首个模型调用所在的帧
        speech = (1000, 2600)                            # 绝对毫秒

        class AbsFunasr:
            """按绝对时间轴报段的假 FSMN：从窗首块 PCM 的首样本还原窗原点。"""

            def __init__(self):
                self.calls = 0

            def generate(self, input=None, cache=None, fs=None):
                self.calls += 1
                head = array.array("h")
                head.frombytes(bytes(input[:2]))
                off = head[0]                            # 窗首块的绝对起始毫秒
                return [{"value": [[max(0, speech[0] - off),
                                    min(window_ms, speech[1] - off)]]}]

        vad = WindowSegmentVad(AbsFunasr(), block_ms=block, window_ms=window_ms,
                               stride_ms=stride_ms, to_float=lambda b: b)

        def truth(i: int) -> float:
            """块 i（绝对 [i*block, (i+1)*block)）是否落在语音内。"""
            return 1.0 if speech[0] < (i + 1) * block and speech[1] > i * block else 0.0

        n = 5000 // block
        probs = [vad.prob(block_pcm(i * block, cfg.block_samples)) for i in range(n)]
        self.assertEqual(len(probs), n, "每帧必须恰好拿到一个概率")
        self.assertTrue(all(p == 0.0 for p in probs[:warm]), "整窗填满前不判决")
        for i in range(warm, n):
            self.assertEqual(probs[i], truth(i - stride_blocks + 1),
                             "第 %d 帧拿到的概率应是第 %d 帧的（恒定滞后 %dms）"
                             % (i, i - stride_blocks + 1, vad.lag_ms))
        self.assertEqual(vad.model_calls, (n - 1 - warm) // stride_blocks + 1, "必须按 stride 批量调模型")
        self.assertLess(vad.model_calls, n / 2, "不得逐帧调模型")
        self.assertEqual(vad.lag_ms, stride_ms - block)
        self.assertGreaterEqual(cfg.vad_lag_ms, vad.lag_ms,
                                "vad_lag 必须盖住批量判决的固有滞后，否则概率与帧时间不一致")

    def test_window_segment_vad_ongoing_tail(self):
        """TC20：段尾贴窗（仍在说话）→ 窗尾概率必须置 1，防误判收尾。"""
        cfg = SegmentConfig(warmup_ms=0)
        block = cfg.block_ms
        window_ms, stride_ms = 1920, 320
        warm = window_ms // block - 1
        margin = 250

        class RelFunasr:
            def __init__(self, end_rel_ms):
                self.end_rel_ms = end_rel_ms
                self.calls = 0

            def generate(self, input=None, cache=None, fs=None):
                self.calls += 1
                return [{"value": [[0, self.end_rel_ms]]}]

        def run(end_rel_ms):
            vad = WindowSegmentVad(RelFunasr(end_rel_ms), block_ms=block, window_ms=window_ms,
                                   stride_ms=stride_ms, ongoing_margin_ms=margin)
            return [vad.prob(b"\x00" * cfg.block_bytes) for _ in range(2 * window_ms // block)]

        ongoing = run(window_ms - 100)      # 段尾距窗尾 100ms < margin → 仍在说话
        self.assertTrue(all(p == 1.0 for p in ongoing[warm:]), "ongoing 时窗尾不能是静音")
        ended = run(window_ms - 400)        # 段尾距窗尾 400ms > margin → 真的说完了
        self.assertEqual(ended[-1], 0.0, "已经超过 margin，窗尾必须落到静音")


# ======================================================================
class TestRingBuffer(unittest.TestCase):
    def test_pow2_capacity_and_wrap(self):
        cfg = SegmentConfig(ring_capacity_ms=8000)
        ring = AudioRingBuffer(cfg)
        self.assertEqual(ring.capacity_ms, 8192, "必须向上取到 2 的幂（131072 样本）")
        for i in range(600):
            ring.write(block_pcm(i, 16), i * 1)      # 每次 1ms，首样本编码写入时刻
        self.assertEqual(ring.w_ms, 600)
        data, status = ring.read(500, 600)
        self.assertIs(status, SampleStatus.CLEAN)
        self.assertEqual(len(data), 100 * cfg.bytes_per_ms)
        a = array.array("h")
        a.frombytes(data[:2])
        self.assertEqual(a[0], 500, "下标映射：读到的首样本必须来自请求时刻")

    def test_clamped_when_reading_evicted_region(self):
        cfg = SegmentConfig(ring_capacity_ms=1024)
        ring = AudioRingBuffer(cfg)
        for i in range(4000):
            ring.write(b"\x00\x00" * 16, i)              # 每次 1ms
        data, status = ring.read(10, 20)
        self.assertIs(status, SampleStatus.CLAMPED)
        self.assertEqual(len(data), 10 * cfg.bytes_per_ms, "长度必须保持，缺的部分补零")

    def test_write_far_ahead_advances_without_drop(self):
        """采集时间轴向前跳（设备重启 / RTC 抖动）：写入成功，W 前进，不算丢弃。"""
        ring = AudioRingBuffer(SegmentConfig(ring_capacity_ms=1024))
        ring.write(b"\x00\x00" * 16, 0)                   # 1ms
        self.assertEqual(ring.write(b"\x00\x00" * 16, 5000), 0)
        self.assertEqual(ring.w_ms, 5001)

    def test_late_frame_is_ignored(self):
        """整帧都在写指针之前 → 忽略，不回退 W（否则会读出撕裂音频）。"""
        ring = AudioRingBuffer(SegmentConfig(ring_capacity_ms=1024))
        for i in range(4000):
            ring.write(b"\x00\x00" * 16, i)
        self.assertEqual(ring.write(b"\x00\x00" * 16, 100), 0)
        self.assertEqual(ring.w_ms, 4000)
        self.assertEqual(ring.dropped_ms, 0)

    def test_oversized_late_frame_reports_dropped_prefix(self):
        """单帧跨过可保留窗口 → 显式上报丢了多少，而不是静默覆盖。"""
        ring = AudioRingBuffer(SegmentConfig(ring_capacity_ms=1024))
        self.assertEqual(ring.write(b"\x00\x00" * 16 * 1500, 0), 476, "比缓冲还长的帧，开头留不下")
        self.assertEqual(ring.w_ms, 1500)
        dropped = ring.write(b"\x00\x00" * 16 * 1300, 300)
        self.assertEqual(dropped, 276)                     # [300,576) 已经读不回来了
        self.assertEqual(ring.w_ms, 1600)
        self.assertEqual(ring.dropped_ms, 752)
        data, status = ring.read(200, 600)
        self.assertIs(status, SampleStatus.CLAMPED, "被覆盖的区间必须显式报出来")
        self.assertEqual(len(data), 400 * ring.bytes_per_ms)


class TestConfig(unittest.TestCase):
    def test_latency_budget_matches_doc(self):
        cfg = SegmentConfig()
        grid = lambda ms: -(-ms // cfg.block_ms) * cfg.block_ms   # 延迟按帧栅格向上取整
        onset, offset, hard = cfg.latency_budget()
        self.assertEqual(onset, cfg.vad_lag_ms + grid(cfg.min_speech_ms))
        self.assertEqual(offset, cfg.vad_lag_ms + grid(cfg.min_silence_ms))
        self.assertEqual(hard, cfg.vad_lag_ms + cfg.max_utterance_ms)
        self.assertEqual((onset, offset, hard), (820, 820, 5500), "文档 §6 的预算表就是这三个数")
        cfg2 = SegmentConfig(emit_partial=True)
        self.assertEqual(cfg2.latency_budget()[1],
                         cfg2.vad_lag_ms + grid(cfg2.long_silence_ms))

    def test_bad_param_combinations_rejected(self):
        for kw in (
            dict(prob_end=0.7, prob_start=0.55),
            dict(long_silence_ms=200, min_silence_ms=300),
            dict(gap_merge_ms=400, min_silence_ms=300),
            dict(max_utterance_ms=400, cut_search_ms=500),
            dict(min_speech_ms=100, confirm_frames=6),
        ):
            cfg = SegmentConfig(**kw)
            with self.assertRaises(ConfigError, msg=str(kw)):
                cfg.validate()

    def test_default_capacity_satisfies_constraint(self):
        cfg = SegmentConfig()
        need = (cfg.max_utterance_ms + cfg.vad_lag_ms + cfg.pre_roll_ms
                + cfg.post_roll_ms + cfg.hard_margin_ms)
        self.assertGreaterEqual(cfg.ring_capacity_ms, need)


class TestStats(unittest.TestCase):
    def test_noise_floor_and_report(self):
        h = Harness()
        h.quiet(3000).speak(1000).quiet(1000)
        rep = h.engine.stats.report()
        self.assertLess(rep["onset_latency_p50"], 1000)
        self.assertEqual(rep["final_segments"], 1)
        self.assertEqual(rep["gap_ms"], 0)
        self.assertLess(h.engine.stats.noise_floor_db(), -50)


def mk_seg(seg_id=1, utt=1, seq=0, s=0, e=1000, reason=CloseReason.MAX_DURATION,
           partial=False, speech=(0, 0), amp=3000, rate=16000, status=SampleStatus.CLEAN):
    """造一片派发段：语音区间内是 ``amp`` 方波、区间外是数字零。

    方波便于校验拼接后的字节序与补零位置，零区间便于校验逐帧电平数组。
    ``status`` 默认 ``clean``，用来模拟采集异常片（文档 §5.4 ``gap`` / ``clamped``）。
    """
    n = (e - s) * rate // 1000
    a = array.array("h", [amp if i % 2 == 0 else -amp for i in range(n)])
    lo, hi = speech
    if hi > lo:
        for i in range(max(0, (lo - s) * rate // 1000)):
            a[i] = 0
        for i in range(min(n, (hi - s) * rate // 1000), n):
            a[i] = 0
    return Segment(
        seg_id=seg_id, utterance_id=utt, seq_in_utterance=seq, start_ms=s, end_ms=e,
        duration_ms=e - s, samples=a.tobytes(), sample_rate=rate,
        close_reason=reason, partial=partial,
        speech_start_ms=lo, last_speech_ms=hi, status=status)


class TestUtteranceAssembly(unittest.TestCase):
    """整句装配层（``collector.py``）：拼接、partial、空洞、裁剪、逐帧数组、交付载荷、队列。"""

    def test_stitches_max_duration_chain(self):
        asm = UtteranceAssembler(tag="t")
        asm.dispatch(mk_seg(1, seq=0, s=0, e=5000, amp=1000))
        asm.dispatch(mk_seg(2, seq=1, s=5000, e=9000, amp=2000, reason=CloseReason.SILENCE))
        utt = asm.take()
        self.assertIsNotNone(utt)
        self.assertEqual((utt.start_ms, utt.end_ms, utt.duration_ms), (0, 9000, 9000))
        self.assertTrue(utt.complete and not utt.aborted)
        self.assertEqual(utt.close_reason, "silence")
        self.assertTrue(utt.contiguous)
        a = utt.to_int16()
        self.assertEqual(len(a), 9000 * 16)
        self.assertEqual({abs(x) for x in a[: 5000 * 16]}, {1000})       # 片顺序没串位
        self.assertEqual({abs(x) for x in a[5000 * 16:]}, {2000})
        self.assertEqual(asm.take(), None)                               # 队列只有一条

    def test_partial_is_preview_only(self):
        asm = UtteranceAssembler()
        # 句子还没开：预览只计数，不会凭空造出一句
        asm.dispatch(mk_seg(1, utt=1, partial=True, s=0, e=3000, reason=CloseReason.SILENCE))
        self.assertIsNone(asm.take())
        self.assertIsNone(asm.current())
        self.assertEqual(asm.stats()["previews"], 1)
        # 句子开了（片 0 是硬截断片）→ 之后的预览挂到在途句，final 才进波形
        asm.dispatch(mk_seg(2, utt=1, seq=0, s=0, e=5000))
        asm.dispatch(mk_seg(3, utt=1, seq=1, partial=True, s=0, e=6000,
                            reason=CloseReason.SILENCE))
        asm.dispatch(mk_seg(3, utt=1, seq=1, s=5000, e=6000, reason=CloseReason.SILENCE))
        utt = asm.take()
        self.assertEqual(utt.duration_ms, 6000)                          # partial 没混进波形
        self.assertEqual([p.seg_id for p in utt.parts], [2, 3])
        self.assertTrue(utt.preview.partial)
        self.assertEqual(utt.preview.seg_id, 3)

    def test_open_utterance_waits_for_non_max_duration(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, s=0, e=5000))
        self.assertIsNone(asm.take())
        cur = asm.current()
        self.assertEqual((cur.utterance_id, len(cur.parts), cur.complete), (1, 1, False))
        pend = asm.close_pending()                                       # 会话结束兜底
        self.assertEqual(len(pend), 1)
        self.assertTrue(pend[0].aborted)
        self.assertEqual(pend[0].close_reason, "session_end")

    def test_hole_is_zero_filled_and_accounted(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, seq=0, s=0, e=1000, amp=3000))
        asm.dispatch(mk_seg(2, seq=1, s=1500, e=2500, amp=3000, reason=CloseReason.EOS))
        utt = asm.take()
        self.assertEqual(utt.duration_ms, 2500)                          # 下标↔毫秒映射没断
        self.assertFalse(utt.contiguous)
        self.assertEqual(utt.hole_ms, 500)
        self.assertEqual([(h.from_seg, h.to_seg, h.gap_ms) for h in utt.holes], [(1, 2, 500)])
        a = utt.to_int16()
        self.assertEqual(set(a[1000 * 16:1500 * 16]), {0})
        self.assertTrue(all(f.zeroed for f in utt.frame_detail(32)
                            if 1000 <= f.start_ms < 1500))
        self.assertFalse(any(f.zeroed for f in utt.frame_detail(32) if f.start_ms >= 1500))

    def test_overlap_head_is_trimmed(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, seq=0, s=0, e=1000))
        asm.dispatch(mk_seg(2, seq=1, s=800, e=1800, reason=CloseReason.EOS))
        utt = asm.take()
        self.assertEqual(utt.duration_ms, 1800)                          # 同一段音频不喂两遍
        self.assertEqual([h.gap_ms for h in utt.holes], [-200])
        self.assertEqual(utt.parts[1].meta["head_trimmed_bytes"], 200 * 16 * 2)

    def test_new_utterance_aborts_pending_one(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, utt=1, s=0, e=5000))                      # 没有收尾片
        asm.dispatch(mk_seg(2, utt=2, seq=0, s=6000, e=7000, reason=CloseReason.EOS))
        out = asm.drain()
        self.assertEqual([u.utterance_id for u in out], [1, 2])
        self.assertTrue(out[0].aborted and not out[1].aborted)

    def test_seq_gap_rejected(self):
        utt = Utterance(1, 16000)
        utt.append(mk_seg(1, seq=0))
        with self.assertRaises(ValueError):
            utt.append(mk_seg(2, seq=2))

    def test_sample_rate_mismatch_rejected(self):
        utt = Utterance(1, 16000)
        with self.assertRaises(ValueError):
            utt.append(mk_seg(1, rate=8000))

    def test_frame_detail_grid_matches_block_ms(self):
        utt = Utterance(1, 16000)
        utt.append(mk_seg(1, s=0, e=1024, reason=CloseReason.EOS, speech=(64, 960)))
        fr = utt.frame_detail(32)
        self.assertEqual(len(fr), 1024 // 32)                            # 与状态机帧数一致
        self.assertEqual([f.index for f in fr[:3]], [0, 1, 2])
        self.assertEqual(fr[0].end_ms - fr[0].start_ms, 32)
        self.assertTrue(any(f.in_speech for f in fr))
        self.assertFalse(fr[0].in_speech)
        self.assertGreater(min(f.rms_db for f in fr if f.in_speech), -30)
        self.assertLess(min(f.rms_db for f in fr if not f.in_speech), -100)

    def test_trim_to_speech_uses_vad_bounds(self):
        utt = Utterance(1, 16000)
        utt.append(mk_seg(1, s=1000, e=3000, reason=CloseReason.EOS, speech=(1200, 2700)))
        trim = utt.trim_to_speech(pad_ms=40)
        self.assertEqual(len(trim), (2700 - 1200 + 80) * 16)             # 语音区间 + 两侧余量
        self.assertEqual(utt.ms_to_index(1200), 1200 * 16 - 1000 * 16)
        self.assertEqual(utt.index_to_ms(16 * 200), 1200)

    # ---------------------------------------------------------------- 交付载荷
    def test_clean_sentence_is_usable_for_all_three_tasks(self):
        """三片硬截断拼回的一句干净话：分离 / ASR / 声纹都该能直接开跑。"""
        asm = UtteranceAssembler(tag="ch0")
        asm.dispatch(mk_seg(1, seq=0, s=0, e=5000, speech=(200, 4800)))
        asm.dispatch(mk_seg(2, seq=1, s=5000, e=10000, speech=(5100, 9800)))
        asm.dispatch(mk_seg(3, seq=2, s=10000, e=12000, reason=CloseReason.SILENCE,
                            speech=(10100, 11800)))
        b = asm.take().to_bundle()
        self.assertIsInstance(b, UtteranceBundle)
        self.assertIsInstance(b.full, AudioView)
        self.assertEqual(b.usable_for, {"separation": "ok", "asr": "ok", "voiceprint": "ok"})
        self.assertTrue(all(b.usable(t) for t in ("separation", "asr", "voiceprint")))
        self.assertFalse(b.suspect)
        self.assertEqual((b.n_parts, b.seq_first, b.seq_last), (3, 0, 2))
        self.assertEqual((b.start_ms, b.end_ms, b.duration_ms), (0, 12000, 12000))
        self.assertEqual(b.close_reason, "silence")
        self.assertEqual((b.speech_start_ms, b.speech_end_ms), (200, 11800))   # 原始 VAD 边界
        self.assertEqual(b.speech.duration_ms, 11800 - 200 + 80)               # 边界 + 两侧 pad
        # 两个视图出自同一次解码：切片能逐字节对上，没有第二次 O(N) 解码
        self.assertEqual(b.speech.pcm, b.full.pcm[160 * 32:11840 * 32])
        import json as _json
        d = _json.loads(b.json())
        self.assertEqual(d["usable_for"]["asr"], "ok")
        self.assertEqual(set(d["views"]), {"full", "speech"})
        self.assertNotIn("samples", d)                                         # 波形不进 JSON
        self.assertNotIn("pcm", d["views"]["full"])

    def test_positive_hole_blocks_separation_but_not_asr(self):
        """正空洞补的是假样本：分离会被它凭空造出一个「源」，转写照常吃剩下的真声。"""
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, seq=0, s=0, e=1000, speech=(100, 900)))
        asm.dispatch(mk_seg(2, seq=1, s=1500, e=2500, reason=CloseReason.EOS,
                            speech=(1600, 2400)))
        b = asm.take().to_bundle()
        self.assertEqual(b.usable_for["separation"], "hole_+500ms")
        self.assertEqual(b.usable_for["asr"], "ok")
        self.assertEqual(b.usable_for["voiceprint"], "hole_+500ms")            # 声纹最怕伪影
        self.assertIn("hole_+500ms", b.suspect_reasons)
        self.assertIn("zeroed_in_speech_500ms", b.suspect_reasons)
        self.assertEqual(b.zeroed_spans_ms, ((1000, 1500),))
        self.assertEqual(b.full.zeroed_spans_ms, ((1000, 1500),))
        self.assertEqual(b.speech.zeroed_spans_ms, ((1000, 1500),))
        self.assertTrue(b.full.has_zeroed(1200, 1300))
        self.assertFalse(b.full.has_zeroed(1500, 2000))
        self.assertEqual(b.speech_duration_ms, 2440 - 60)

    def test_overlap_is_trimmed_and_flagged_as_suspect(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, seq=0, s=0, e=1000, speech=(100, 900)))
        asm.dispatch(mk_seg(2, seq=1, s=800, e=2800, reason=CloseReason.EOS,
                            speech=(900, 2700)))
        b = asm.take().to_bundle()
        self.assertIn("overlap_200ms", b.suspect_reasons)
        self.assertEqual(b.usable_for["separation"], "ok")                     # 裁头即可，没造假样本
        self.assertEqual(b.usable_for["voiceprint"], "overlap_200ms")
        self.assertEqual(b.hole_ms, 200)

    def test_short_speech_is_not_enough_for_voiceprint(self):
        """短音频的说话人嵌入极不稳：ASR 照转，声纹要够长才放行。"""
        utt = Utterance(1, 16000)
        utt.append(mk_seg(1, s=1000, e=3000, reason=CloseReason.EOS, speech=(1200, 1500)))
        b = utt.to_bundle()
        self.assertEqual(b.speech_duration_ms, 300 + 80)
        self.assertEqual(b.usable_for["voiceprint"], "short_380ms")
        self.assertEqual(b.usable_for["asr"], "ok")
        self.assertEqual(utt.to_bundle(min_voiceprint_ms=300).usable_for["voiceprint"], "ok")
    def test_bundle_without_vad_speech_has_empty_speech_view(self):
        """没判出语音时 speech 视图是空的（不退回整句），下游看时长就知道该跳过。

        在途句子的打包只在单线程里读（真实链路要等 ``take()``，见 ``to_bundle`` 文档）。
        """
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, utt=1, s=0, e=5000))                            # 没有收尾片
        live = asm.current().to_bundle()                                       # 句子还在说
        self.assertIn("incomplete", live.suspect_reasons)
        self.assertEqual(live.usable_for["separation"], "ok")
        b = asm.close_pending()[0].to_bundle()                                 # 兜底强制收尾
        self.assertTrue(b.aborted)
        self.assertEqual(b.close_reason, "session_end")
        self.assertIn("aborted", b.suspect_reasons)
        self.assertEqual(b.speech.n_samples, 0)
        self.assertEqual((b.speech_start_ms, b.speech_end_ms, b.speech_duration_ms), (0, 0, 0))
        self.assertEqual(b.usable_for["asr"], "no_speech")
        self.assertEqual(b.usable_for["voiceprint"], "short_0ms")
        self.assertEqual(b.usable_for["separation"], "ok")                     # 波形本身是完整的
        self.assertFalse(b.usable("asr"))

    def test_dirty_segment_status_makes_the_bundle_suspect(self):
        """片本身带 gap（采集 overrun / E7 强制闭合）时必须挂到 suspect 上。"""
        utt = Utterance(1, 16000)
        utt.append(mk_seg(1, s=1000, e=4000, reason=CloseReason.SILENCE,
                          speech=(1100, 3800), status=SampleStatus.GAP))
        b = utt.to_bundle()
        self.assertIn("seg#1_gap", b.suspect_reasons)
        self.assertEqual(b.usable_for["voiceprint"], "seg#1_gap")
        self.assertEqual(b.usable_for["asr"], "ok")
        self.assertEqual(b.usable_for["separation"], "ok")

    def test_view_time_mapping_slicing_and_numpy_scale(self):
        asm = UtteranceAssembler()
        asm.dispatch(mk_seg(1, s=1000, e=2000, reason=CloseReason.EOS, speech=(1032, 1900)))
        b = asm.take().to_bundle(frame_ms=32, with_frames=True)
        v = b.full
        self.assertEqual(v.index_to_ms(v.ms_to_index(1500)), 1500)              # 下标 ↔ 毫秒互逆
        self.assertEqual(v.ms_to_index(999999), v.n_samples)                    # 越界夹住而不是报错
        self.assertEqual(v.level_db()[1], 20.0 * math.log10(3000 / 32768.0))
        w = v.slice_ms(1200, 1400)
        self.assertEqual((w.source, w.start_ms, w.duration_ms), ("full[1200,1400]", 1200, 200))
        self.assertEqual(len(w), 200 * 16)
        with self.assertRaises(ValueError):
            v.slice_ms(5000, 6000)                                              # 越界切片不静默给空
        with self.assertRaises(ValueError):
            b.audio("left")
        self.assertEqual(len(b.frames), -(-1000 // 32))                        # 末帧不满也计一帧
        self.assertTrue(any(f.in_speech for f in b.frames))
        try:
            import numpy as np
        except ImportError:
            return                                                             # 核心链路不依赖 numpy
        a = v.to_numpy()
        self.assertEqual(a.shape, (1000 * 16,))
        self.assertEqual(str(a.dtype), "float32")
        self.assertAlmostEqual(float(np.max(np.abs(a))), 3000 / 32768.0, places=5)
        self.assertEqual(int(np.max(np.abs(v.to_numpy(normalize=False)))), 3000)

    def test_queue_overflow_drops_oldest(self):
        asm = UtteranceAssembler(max_queued=2)
        for i in range(1, 5):
            asm.dispatch(mk_seg(i, utt=i, s=0, e=1000, reason=CloseReason.EOS))
        self.assertEqual(asm.stats()["dropped_queued"], 2)
        self.assertEqual([u.utterance_id for u in asm.drain()], [3, 4])

    def test_broadcast_fans_out_and_propagates_refusal(self):
        seen = []
        asm = UtteranceAssembler()
        bc = BroadcastDispatcher([asm, CallbackDispatcher(seen.append)])
        self.assertTrue(bc.dispatch(mk_seg(1, reason=CloseReason.EOS)))
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(asm.take())
        # 任一下游拒收就要告诉状态机「没吃下」，否则背压统计会说谎
        q = QueueDispatcher(maxsize=1, drop_policy="reject")
        bc2 = BroadcastDispatcher([asm, q])
        self.assertTrue(bc2.dispatch(mk_seg(2, utt=2, reason=CloseReason.EOS)))
        self.assertFalse(bc2.dispatch(mk_seg(3, utt=3, reason=CloseReason.EOS)))
        self.assertEqual(q.stats()["dropped"], 1)
        self.assertEqual(asm.stats()["parts"], 3)

    def test_writer_emits_wav_and_manifest_with_frames(self):
        import json as _json
        import shutil
        import tempfile

        out = tempfile.mkdtemp(prefix="seg_utt_")
        try:
            asm = UtteranceAssembler(tag="ch0")
            writer = UtteranceWriter(out, frames=True, frame_ms=32)
            asm.dispatch(mk_seg(1, seq=0, s=0, e=5000, speech=(128, 4900)))
            asm.dispatch(mk_seg(2, seq=1, s=5000, e=6000, reason=CloseReason.SILENCE,
                                speech=(5024, 5900)))
            utt = asm.take()
            rec = writer(utt)
            self.assertEqual(rec["n_samples"], 6000 * 16)
            self.assertEqual(len(rec["frames"]), -(-6000 // 32))          # 末帧不满也计一帧
            for kind in ("full", "trim"):
                with wave.open(rec["files"][kind], "rb") as r:
                    self.assertEqual((r.getnchannels(), r.getsampwidth(), r.getframerate()),
                                     (1, 2, 16000))
                    self.assertEqual(r.getnframes(), rec["n_samples"] if kind == "full"
                                     else rec["trim_samples"])
            with open(os.path.join(out, "ch0", "manifest.jsonl"), encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(_json.loads(lines[0])["utterance_id"], 1)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_engine_to_assembler_end_to_end(self):
        """接在真状态机出口上：6.2s 连续语音 → 硬截断两片 → 装配回一整句。"""
        asm = UtteranceAssembler(tag="live")
        seen = []
        h = Harness(dispatcher=BroadcastDispatcher([asm, CallbackDispatcher(seen.append)]))
        h.speak(6200).quiet(700)
        h.engine.flush()
        utt = asm.take()
        self.assertIsNotNone(utt, "dispatch 之后整句应已可 take()")
        self.assertGreaterEqual(len(utt.parts), 2)                       # 确实被硬截断过
        self.assertGreaterEqual(len(seen), 2)                            # 广播没吞掉任何一片
        self.assertTrue(utt.complete and not utt.aborted)
        self.assertEqual(utt.hole_ms, 0)
        self.assertEqual(utt.duration_ms, utt.end_ms - utt.start_ms)
        self.assertGreaterEqual(utt.duration_ms, 6200)
        # PCM 来自环形缓冲：每块首样本编码了该块的起始毫秒（见文件 docstring）
        self.assertEqual(utt.to_int16()[0], utt.start_ms)
        b = utt.to_bundle()                                                    # 交付载荷走真链路
        self.assertTrue(b.usable("separation") and b.usable("asr"))
        self.assertEqual(b.suspect_reasons, ())
        self.assertGreaterEqual(b.speech_duration_ms, 5000)                     # VAD 边界没吞内容
        self.assertEqual(asm.close_pending(), [])                        # 没有半截在途句


def _tagged_pcm(tag: int, frames: int = 160) -> bytes:
    """首样本写「块编号」的 PCM，便于断言状态机到底吃到了哪几块。"""
    a = array.array("h", [0]) * frames
    a[0] = tag
    return a.tobytes()


class _Recorder:
    """替 ``pump()`` 记它到底喂了哪些块、有没有被要求跳时钟。"""

    def __init__(self) -> None:
        self.tags = []
        self.skips = []
        self.dispatched = 0
        self.total_ms = 0

    def feed(self, a) -> None:
        self.tags.append(a[0])

    def skip(self, frames: int) -> None:
        self.skips.append(frames)


class _BacklogSource:
    """起流后、``pump()`` 开吃前已经攒了 3 块旧数据（tag 1..3），之后才是新数据（101..）。"""

    realtime, channels, rate, device_name = True, 1, 16000, "backlog-src"

    def __init__(self, with_drain: bool = True) -> None:
        self._pending = [_tagged_pcm(i) for i in (1, 2, 3)]     # 模型加载期间的积压
        self._fresh = [_tagged_pcm(i) for i in (101, 102)]
        self.dropped_frames = 3 * 160                           # 队列满，回调丢了 3 块
        self.overruns = 1
        self.startup_dropped = 0
        if with_drain:                                          # 第三方源可以不实现它
            self.drain_pending = self._drain

    def __enter__(self) -> "_BacklogSource":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def _drain(self):
        n = len(self._pending)
        self._pending = []
        self.startup_dropped += self.dropped_frames
        self.dropped_frames = self.overruns = 0
        return n, n * 160

    def blocks(self):
        for pcm in self._pending + self._fresh:
            yield pcm, 160


class TestCaptureDrain(unittest.TestCase):
    """采集侧接缝（demo2）：起流积压必须在第一个块之前被丢掉（文档 §9.2）。"""

    def _pump(self, src):
        import argparse

        from com.robot.segment.demo2 import pump

        rec = _Recorder()
        args = argparse.Namespace(quiet=True, no_heartbeat=True, duration=0)
        stats = pump(src, [rec], None, args)
        return rec, stats

    def test_pump_discards_startup_backlog(self):
        src = _BacklogSource()
        rec, stats = self._pump(src)
        self.assertEqual(rec.tags, [101, 102])                   # 只处理当前时刻之后的块
        self.assertEqual(rec.skips, [])                          # 积压没被当成会话内空洞
        self.assertEqual((src.dropped_frames, src.overruns), (0, 0))
        self.assertEqual(src.startup_dropped, 480)               # 起流那部分另记一笔
        self.assertEqual(stats["frames"], 320)

    def test_source_without_drain_pending_still_works(self):
        src = _BacklogSource(with_drain=False)
        rec, _ = self._pump(src)
        self.assertEqual(rec.tags, [1, 2, 3, 101, 102])          # 最小契约只有 blocks()
        self.assertEqual(src.startup_dropped, 0)                 # 只是失去这层保护

    def test_mic_drain_clears_queue_and_moves_counters(self):
        import queue

        from com.robot.segment.demo2 import MicSource

        ms = MicSource.__new__(MicSource)                        # 绕开 sounddevice，只测队列语义
        ms._q = queue.Queue()
        for _ in range(3):
            ms._q.put((_tagged_pcm(0), 160))
        ms.dropped_frames, ms.overruns, ms.startup_dropped = 500, 2, 0
        self.assertEqual(ms.drain_pending(), (3, 480))
        self.assertTrue(ms._q.empty())
        self.assertEqual((ms.dropped_frames, ms.overruns), (0, 0))
        self.assertEqual(ms.startup_dropped, 480)


def main() -> int:
    import logging

    logging.basicConfig(level=logging.CRITICAL)
    return 0 if unittest.main(verbosity=2, exit=False).result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
