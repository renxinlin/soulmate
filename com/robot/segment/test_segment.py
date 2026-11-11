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
"""
from __future__ import annotations

import array
import os
import sys
import unittest

if __package__ in (None, ""):  # 直接以脚本方式运行时补 sys.path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from com.robot.segment.config import ConfigError, SegmentConfig
from com.robot.segment.dispatcher import CallbackDispatcher, QueueDispatcher
from com.robot.segment.engine import SegmentEngine
from com.robot.segment.gate import StaticGate
from com.robot.segment.models import AudioFrame, CloseReason, SampleStatus, SegState
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

def main() -> int:
    import logging

    logging.basicConfig(level=logging.CRITICAL)
    return 0 if unittest.main(verbosity=2, exit=False).result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
