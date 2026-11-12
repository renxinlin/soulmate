# -*- coding: utf-8 -*-
"""实时采集 demo2：``麦克风 → 通道处理 → 16k 重采样 → 逐帧 VAD → 分段状态机``。

与 ``demo.py``（离线回放一条 16k/mono wav）的差别：这里没有现成的整文件，
音频是设备持续推来的块，采样率/通道数由设备决定，所以本 demo 额外负责三件
属于「采集侧」的事（设计文档 §5.1 / §7-G12）：

1. **通道数**：状态机只吃单声道（``SegmentConfig.validate`` 强制 ``channels == 1``），
   多通道必须先落成单路：

   * ``--mix mean``   线性平均下混（最稳，阵列同相时用）
   * ``--mix energy`` 逐块选电平最高的通道（近讲跟随）。带 ``keep_db`` 滞回 +
     跨块淡入淡出，避免在语音中间硬切导致相位反转
   * ``--channel N``  固定取第 N 路
   * ``--mix multi``  **每个通道一套独立状态机**（双麦/阵列各自分段，
     谁先说完谁先送 ASR，互不等待）
2. **采样率**：设备常见 44.1k/48k，而判决与模型都按 16k。整数比走箱式平均抽取
   （带抗混叠），非整数比走跨块连续的线性插值。
3. **音频时钟**：状态机的时间坐标是「已交付样本数」换算的单调毫秒，
   **不是墙钟、也不是缓冲区下标**。采集端丢块时故意把时钟推前，让状态机
   看见真实空洞并触发 ``gap_force_close``（文档 §7-G12），
   而不是把丢掉的音频假装没发生过。

用法::

    python -m com.robot.segment.demo2 --list-devices
    python -m com.robot.segment.demo2                            # 默认麦克风 + 能量 VAD
    python -m com.robot.segment.demo2 --channels 2 --mix energy --out /tmp/seg
    python -m com.robot.segment.demo2 --channels max --mix multi --duration 30
    python -m com.robot.segment.demo2 --vad sherpa --silero-model /path/silero_vad.onnx
    python -m com.robot.segment.demo2 --input wav 48k_2ch.wav    # 真录音走同一条实时链路
    python -m com.robot.segment.demo2 --input sim --channels 4   # 无麦克风容器/CI 自测

按 Ctrl-C 停止采集，会把在途段闭合（EOS）后打印完整报告。
麦克风依赖 ``sounddevice``（PortAudio）；要换 ``pyaudio`` 只需再写一个产出
``(interleaved_int16_bytes, frames)`` 的 source，其余链路不变。
"""
from __future__ import annotations

import argparse
import array
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import wave
from typing import Iterator, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # 直接以脚本方式运行时补 sys.path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from com.robot.segment.config import SegmentConfig
from com.robot.segment.demo import report, write_wav
from com.robot.segment.dispatcher import CallbackDispatcher
from com.robot.segment.engine import SegmentEngine
from com.robot.segment.models import SegState, Segment
from com.robot.segment.ring import AudioRingBuffer
from com.robot.segment.vad import EnergyVad, WindowSegmentVad, rms_db

logger = logging.getLogger(__name__)


def _clip(x: int) -> int:
    return -32768 if x < -32768 else (32767 if x > 32767 else x)


# ======================================================================
# 重采样
# ======================================================================
class Resampler:
    """跨块连续的单声道重采样。

    * 整数比（48k→16k 等）用**箱式平均抽取**：先对 ``k`` 个样本求均值再丢弃，
      等效短窗低通，避免直接丢弃产生镜像频谱。
    * 非整数比（44.1k→16k）用相位累加线性插值，块间保留 1 个样本历史，接缝不跳变。
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src, self.dst = src_rate, dst_rate
        self.out_frames = 0
        self._int_ratio = dst_rate > 0 and src_rate % dst_rate == 0
        self._k = src_rate // dst_rate if self._int_ratio else 0
        self._acc = 0
        self._cnt = 0
        self._phase = 0.0
        self._hist: array.array = array.array("h")

    def process(self, a: array.array) -> array.array:
        if self.src == self.dst:
            self.out_frames += len(a)
            return a
        if not a:
            return array.array("h")
        if self._int_ratio:
            k = self._k
            out = array.array("h")
            acc, cnt = self._acc, self._cnt
            for v in a:
                acc += v
                cnt += 1
                if cnt == k:
                    out.append(_clip(acc // k))
                    acc, cnt = 0, 0
            self._acc, self._cnt = acc, cnt
            self.out_frames += len(out)
            return out
        xs = (self._hist + a) if len(self._hist) else a
        step = self.src / self.dst
        n = len(xs)
        out = array.array("h")
        while self._phase < n - 1:
            i = int(self._phase)
            f = self._phase - i
            out.append(_clip(int(xs[i] * (1.0 - f) + xs[i + 1] * f)))
            self._phase += step
        self._phase -= max(0, n - 1)
        self._hist = a[-1:]
        self.out_frames += len(out)
        return out


# ======================================================================
# 多通道 → 单通道
# ======================================================================
class Downmix:
    """把 ``nch`` 路单声道块合成一路（``--mix multi`` 时不走这里）。"""

    def __init__(self, nch: int, mode: str = "mean", channel: int = -1,
                 keep_db: float = 3.0, fade_blocks: int = 2) -> None:
        if channel >= 0:
            if channel >= nch:
                raise ValueError("--channel %d 超出通道数 %d" % (channel, nch))
            mode = "copy"
        if mode not in ("mean", "energy", "copy"):
            raise ValueError("mix 只能是 mean|energy|copy，收到 %r" % mode)
        self.nch = max(1, nch)
        self.mode = mode
        self.pick = max(0, channel)
        self.keep_db = keep_db
        self.fade_blocks = max(1, fade_blocks)
        self.sel = self.pick
        self.switches = 0
        self.levels: List[float] = []
        self._fade_left = 0
        self._fade_from: Optional[array.array] = None
        self._fade_to: Optional[array.array] = None

    def process(self, chans: Sequence[array.array]) -> array.array:
        if self.nch == 1 and self.mode == "mean":
            return chans[0]
        self.levels = [rms_db(c.tobytes()) for c in chans]
        if self.mode == "mean":
            ref = chans[0]
            acc = array.array("h", bytes(len(ref) * 2))
            for i in range(len(ref)):
                acc[i] = _clip(sum(int(c[i]) for c in chans) // self.nch)
            return acc
        if self.mode == "copy":
            return chans[self.pick]
        # energy：只有明显更强才切换，切换处按块线性交叉淡化
        best = max(range(len(self.levels)), key=lambda i: self.levels[i])
        if best != self.sel and self.levels[best] > self.levels[self.sel] + self.keep_db:
            self._fade_from, self._fade_to = chans[self.sel], chans[best]
            self._fade_left = self.fade_blocks
            self.sel = best
            self.switches += 1
        if self._fade_left and self._fade_from is not None and self._fade_to is not None:
            g = 1.0 - self._fade_left / (self.fade_blocks + 1.0)
            ref = self._fade_to
            out = array.array("h", bytes(len(ref) * 2))
            for i in range(len(ref)):
                out[i] = _clip(int(self._fade_from[i] * (1.0 - g) + ref[i] * g))
            self._fade_left -= 1
            if self._fade_left == 0:
                self._fade_from = self._fade_to = None
            return out
        return chans[self.sel]


# ======================================================================
# 一路完整管线
# ======================================================================
class Segmenter:
    """一路 mono 采集流 → 16k → 帧栅格 → :class:`SegmentEngine`。"""

    def __init__(self, cfg: SegmentConfig, src_rate: int, vad, name: str,
                 live: bool = True) -> None:
        self.cfg = cfg
        self.name = name
        self.src_rate = src_rate
        self.resampler = Resampler(src_rate, cfg.sample_rate)
        self.vad = vad
        self.ring = AudioRingBuffer(cfg)
        self.segments: List[Segment] = []
        self.engine = SegmentEngine(
            cfg, self.ring, CallbackDispatcher(self._on_segment),
            on_state_change=self._on_state if live else None,
        )
        self._carry = array.array("h")
        self.delivered = 0       # 已交给状态机的 16k 样本数（音频时钟基准）
        self.dropped_16k = 0     # 采集侧丢弃、只推时钟不进状态机的样本数

    # ------------------------------------------------------------------
    def _on_segment(self, seg: Segment) -> None:
        self.segments.append(seg)
        print("[%s] %s" % (self.name, seg.summary()), flush=True)

    def _on_state(self, old: SegState, new: SegState) -> None:
        print("[%s] %s→%s @ C=%dms" % (self.name, old.value, new.value,
                                       self.engine.c_ms), flush=True)

    def feed(self, mono: array.array) -> None:
        out = self.resampler.process(mono)
        if len(out):
            self._carry.extend(out)
        bs = self.cfg.block_samples
        while len(self._carry) >= bs:
            blk = self._carry[:bs]
            del self._carry[:bs]
            start_ms = self.delivered * 1000 // self.cfg.sample_rate
            raw = blk.tobytes()
            self.ring.write(raw, start_ms)          # 段音频的唯一来源（文档 §2）
            self.engine.feed(raw, start_ms, self.vad)
            self.delivered += bs

    def skip(self, src_frames: int) -> None:
        """采集丢块：把音频时钟推前但**不**喂数据，让状态机看见空洞（§7-G12）。"""
        if src_frames <= 0:
            return
        n = int(src_frames * self.cfg.sample_rate / self.src_rate)
        self._carry = array.array("h")
        self.delivered += n
        self.dropped_16k += n
        print("[%s] 采集丢弃 %d 帧（%.0fms）→ 音频时钟推前，等待空洞强制闭合"
              % (self.name, src_frames, n * 1000.0 / self.cfg.sample_rate), flush=True)

    def flush(self) -> Optional[Segment]:
        return self.engine.flush()

    @property
    def total_ms(self) -> int:
        return self.delivered * 1000 // self.cfg.sample_rate


# ======================================================================
# 采集源：统一产出「交织 int16 字节 + 帧数」
# ======================================================================
class MicSource:
    """``sounddevice`` 输入流；``--channels max`` / ``--rate 0`` 用设备默认值。"""

    def __init__(self, args: argparse.Namespace) -> None:
        import sounddevice as sd  # 惰性导入：headless 环境仍能跑 --input wav/sim

        self.dropped_frames = 0
        self.overruns = 0
        self.realtime = True
        self._q: queue.Queue = queue.Queue(maxsize=max(2, args.qsize))
        self._stop = threading.Event()

        devs = _input_devices(sd)
        dev = _pick_device(args.input_device, devs, sd)
        info = sd.query_devices(dev, "input")
        ch = (int(info["max_input_channels"]) if args.channels == "max"
              else int(args.channels))
        if args.channels != "max" and ch > int(info["max_input_channels"]):
            raise ValueError("设备 %r 最多 %d 个输入通道，请求 %d（可用 --channels max）"
                             % (info["name"], info["max_input_channels"], ch))
        rate = int(args.rate) or int(info["default_samplerate"])
        if ch < 1 or rate <= 0:
            raise ValueError("设备 %r 没有可用输入（channels=%d rate=%d）；"
                             "headless 环境请用 --input sim 或 --input wav"
                             % (info["name"], ch, rate))
        self.channels, self.rate, self.device_name = ch, rate, info["name"]
        blocksize = int(args.blocksize) or max(1, rate // 100)      # 默认 10ms

        def callback(indata, frames, time_info, status):            # PortAudio 线程
            if status:
                self.overruns += 1
            try:
                self._q.put((indata.tobytes(), int(frames)), timeout=0.5)
            except queue.Full:
                self.dropped_frames += int(frames)

        self._stream = sd.InputStream(samplerate=rate, channels=ch, blocksize=blocksize,
                                      dtype="int16", callback=callback, device=dev)
        self.blocksize = blocksize

    def __enter__(self) -> "MicSource":
        self._stream.start()
        return self

    def blocks(self) -> Iterator[Tuple[bytes, int]]:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            yield item

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._stream.stop()
        self._stream.close()


def _input_devices(sd) -> List[Tuple[int, dict]]:
    """列出可用的输入设备；无设备（容器/headless）时给出可执行的提示。"""
    try:
        out = [(i, d) for i, d in enumerate(sd.query_devices())
               if int(d.get("max_input_channels", 0)) > 0]
    except Exception as exc:            # PortAudio 在完全无设备时会直接抛错
        raise ValueError("查询音频设备失败：%s。headless/容器里通常没有麦克风，"
                         "可用 --input sim（合成多通道流）或 --input wav（真录音走同一条"
                         "实时链路）自测；真机请确认 arecord -l 有设备。" % exc)
    if not out:
        raise ValueError("没有带输入通道的音频设备。用 --list-devices 查看全部设备，"
                         "或改用 --input sim / --input wav 自测。")
    return out


def _pick_device(want: str, devs: List[Tuple[int, dict]], sd) -> int:
    if want in ("", "default"):
        dflt = sd.default.device[0]
        return dflt if dflt in [i for i, _ in devs] else devs[0][0]
    if want.isdigit():
        idx = int(want)
        if idx not in [i for i, _ in devs]:
            raise ValueError("设备序号 %d 不是输入设备；可选 %s" % (idx, [i for i, _ in devs]))
        return idx
    for i, d in devs:
        if want.lower() in str(d["name"]).lower():
            return i
    raise ValueError("--input-device %r 没匹配到；可用输入设备：%s"
                     % (want, [d["name"] for _, d in devs]))


class WavSource:
    """把 wav 当采集流喂：走的是与麦克风完全相同的重采样/下混/时钟路径。"""

    def __init__(self, path: str, blocksize: int = 480) -> None:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("%s: 需要 16bit PCM，实际 %d 字节" % (path, w.getsampwidth()))
            self.rate = w.getframerate()
            self.channels = w.getnchannels()
            self._pcm = w.readframes(w.getnframes())
        self.blocksize = max(1, int(blocksize) or self.rate // 100)
        self.dropped_frames = 0
        self.overruns = 0
        self.realtime = False            # 文件回放：尽快跑完整条链路，不等墙钟
        self.device_name = os.path.basename(path)

    def __enter__(self) -> "WavSource":
        return self

    def blocks(self) -> Iterator[Tuple[bytes, int]]:
        step = self.blocksize * self.channels * 2
        for off in range(0, len(self._pcm) - len(self._pcm) % step, step):
            chunk = self._pcm[off:off + step]
            yield chunk, len(chunk) // (self.channels * 2)

    def __exit__(self, *exc) -> None:
        pass


class SimSource:
    """无麦克风环境（容器/CI）的合成多通道流：48k、``nch`` 路交织 int16。

    场景刻意覆盖设计文档的几条判决：短促音（应被 ``min_speech`` 丢弃）、
    正常两句（silence 闭合）、一句超过 ``max_utterance``（硬截断），
    并且**语音只出现在部分通道**，用来验证 ``--mix energy`` 选通道与 ``--mix multi``
    各通道独立分段。
    """

    def __init__(self, channels: int = 2, duration_s: float = 14.0, rate: int = 48000,
                 blocksize: int = 480) -> None:
        self.rate = rate
        self.channels = max(1, int(channels))
        self.blocksize = max(1, int(blocksize) or rate // 100)
        self.total = int(duration_s * rate)
        self.dropped_frames = 0
        self.overruns = 0
        self.realtime = False
        self.device_name = "sim(%dch@%dHz)" % (self.channels, rate)
        self._i = 0
        self._seed = 12345

    # 事件表：(起始秒, 时长秒, 幅度, 通道)
    EVENTS = ((0.6, 0.15, 3000, 0),      # 150ms 短促音 → 应被 min_speech 丢弃
              (1.4, 1.0, 9000, 1),       # 正常句 1，只在 ch1（ louder）
              (3.2, 0.9, 7000, 0),       # 正常句 1 的另一麦（电平更低）
              (5.0, 6.0, 8000, 0),       # 6s 长句 → 触发 max_utterance 截断
              (12.0, 1.0, 9000, 1))      # 正常句 2

    def _rnd(self) -> float:
        self._seed = (1103515245 * self._seed + 12345) & 0x7FFFFFFF
        return self._seed / 0x7FFFFFFF * 2.0 - 1.0

    def __enter__(self) -> "SimSource":
        return self

    def blocks(self) -> Iterator[Tuple[bytes, int]]:
        while self._i < self.total:
            n = min(self.blocksize, self.total - self._i)
            chans: List[array.array] = [array.array("h", bytes(n * 2)) for _ in range(self.channels)]
            for j in range(n):
                t = (self._i + j) / float(self.rate)
                for c in range(self.channels):
                    v = 120.0 * self._rnd()                      # 底噪
                    for s, d, amp, ec in self.EVENTS:
                        if ec != c or not (s <= t < s + d):
                            continue
                        tone = math.sin(2 * math.pi * 300.0 * t) + 0.4 * math.sin(2 * math.pi * 780.0 * t)
                        env = min(1.0, (t - s) / 0.03) * min(1.0, (s + d - t) / 0.05)
                        v += amp * tone * env
                    chans[c][j] = _clip(int(v))
            merged = array.array("h")
            for j in range(n):
                for c in range(self.channels):
                    merged.append(chans[c][j])
            self._i += n
            yield merged.tobytes(), n

    def __exit__(self, *exc) -> None:
        pass


# ======================================================================
# VAD
# ======================================================================
def build_vad(args: argparse.Namespace, cfg: SegmentConfig):
    if args.vad == "energy":
        return EnergyVad(threshold_db=args.threshold_db)
    if args.vad == "sherpa":
        from com.robot.segment.vad import SherpaSileroVad  # 惰性：可选依赖

        vad = SherpaSileroVad(args.silero_model, sample_rate=cfg.sample_rate,
                              threshold=args.prob_start)
        print("[vad] sherpa-onnx silero（逐帧二值判决），模型=%s" % args.silero_model)
        return vad
    from funasr import AutoModel                     # 惰性：只有真用 FSMN 才依赖

    model = AutoModel(model=args.funasr_model, device=args.torch_device)
    vad = WindowSegmentVad(model, sample_rate=cfg.sample_rate, block_ms=cfg.block_ms,
                           window_ms=args.window_ms, stride_ms=args.stride_ms)
    print("[vad] FSMN 批量判决：window=%dms stride=%dms 固有滞后=%dms（配置 vad_lag=%dms）"
          % (vad.window_ms, vad.stride_ms, vad.lag_ms, cfg.vad_lag_ms))
    if cfg.vad_lag_ms < vad.lag_ms:
        print("[warn] vad_lag_ms(%d) < 批量判决滞后(%dms)，概率流与帧时间不一致"
              % (cfg.vad_lag_ms, vad.lag_ms))
    return vad


# ======================================================================
# 主流程
# ======================================================================
def make_pipelines(args: argparse.Namespace, cfg: SegmentConfig, src) -> Tuple[
        List[Segmenter], Optional[Downmix]]:
    nch = src.channels
    if args.mix == "multi" and nch > 1 and args.vad != "energy":
        print("[note] --mix multi 每个通道一套独立 VAD 实例（%d × %s），"
              "内存与算力按通道数线性增长" % (nch, args.vad))
    if args.mix == "multi":
        segs = [Segmenter(cfg, src.rate, build_vad(args, cfg), "ch%d" % c,
                          live=not args.quiet) for c in range(nch)]
        return segs, None
    mix = "mean" if nch == 1 and args.channel < 0 else args.mix
    dm = Downmix(nch, mode=mix, channel=args.channel, keep_db=args.keep_db,
                 fade_blocks=args.fade_blocks)
    name = {"mean": "mix:mean", "energy": "mix:energy", "copy": "mix:ch%d" % dm.pick}[dm.mode]
    return [Segmenter(cfg, src.rate, build_vad(args, cfg), name, live=not args.quiet)], dm


def run(args: argparse.Namespace) -> int:
    cfg = SegmentConfig(
        sample_rate=args.sample_rate, channels=1, block_ms=args.block_ms,
        ring_capacity_ms=args.ring_ms, vad_lag_ms=args.vad_lag,
        min_speech_ms=args.min_speech, min_silence_ms=args.min_silence,
        long_silence_ms=args.long_silence, max_utterance_ms=args.max_utterance,
        pre_roll_ms=args.pre_roll, post_roll_ms=args.post_roll,
        prob_start=args.prob_start, prob_end=args.prob_end,
        emit_partial=args.emit_partial, warmup_ms=args.warmup,
    )
    cfg.validate()

    src = _make_source(args)
    with src:
        segs, dm = make_pipelines(args, cfg, src)
        print("输入：%s @ %dHz / %d 通道 → 目标 %dHz，%s" % (
            src.device_name, src.rate, src.channels, cfg.sample_rate,
            ("每通道独立状态机（%d 套）" % len(segs)) if dm is None else
            ("下混：%s（滞回 %.1fdB，淡化 %d 块）" % (dm.mode, dm.keep_db, dm.fade_blocks))))
        print("参数：block=%dms vad_lag=%dms min_speech=%dms min_silence=%dms "
              "max_utter=%dms pre=%dms post=%dms emit_partial=%s" % (
                  cfg.block_ms, cfg.vad_lag_ms, cfg.min_speech_ms, cfg.min_silence_ms,
                  cfg.max_utterance_ms, cfg.pre_roll_ms, cfg.post_roll_ms, cfg.emit_partial))

        nch = src.channels
        seen_drops = getattr(src, "dropped_frames", 0)
        t0 = time.monotonic()
        last_beat = 0.0
        frames_total = 0
        for pcm, frames in src.blocks():
            frames_total += frames
            lost = getattr(src, "dropped_frames", 0) - seen_drops
            if lost > 0:
                for s in segs:
                    s.skip(lost)
                seen_drops = getattr(src, "dropped_frames", 0)
            a = array.array("h")
            a.frombytes(pcm[:len(pcm) - len(pcm) % 2])
            if nch == 1:
                chans = [a]
            else:
                chans = [a[c::nch] for c in range(nch)]
            if dm is None:                       # multi：每通道一套状态机
                for c, s in enumerate(segs):
                    s.feed(chans[c])
            else:
                segs[0].feed(dm.process(chans))

            now = time.monotonic()
            if (src.realtime and not args.quiet and not args.no_heartbeat
                    and now - last_beat >= 1.0):
                last_beat = now
                aud = segs[0].total_ms
                print("\r[采集 %.1fs] 音频时钟 %.2fs（漂移 %+.0fms）段数=%d        "
                      % (now - t0, aud / 1000.0, aud - (now - t0) * 1000.0,
                         sum(len(s.segments) for s in segs)), flush=True)
            if src.realtime and args.duration and (now - t0) >= args.duration:
                print("\n到达 --duration %.1fs（墙钟），停止采集" % args.duration, flush=True)
                break

    for s in segs:                               # §7-G15：停止采集必须闭合在途段
        s.flush()

    total_ms = max([s.total_ms for s in segs] or [0])
    if not args.quiet:
        for s in segs:
            print("\n" + "=" * 20 + " %s " % s.name + "=" * 20)
            report(s.segments, s.engine, s.ring, total_ms, width=args.width)
        if dm is not None:
            print("\n通道选择：当前=%s 切换次数=%d 末块电平=%s" % (
                dm.sel, dm.switches,
                ["%.0fdB" % v for v in dm.levels] if dm.levels else "n/a"))
    print("\n采集帧=%d（%.2fs）丢弃=%d 次 overruns=%d" % (
        frames_total, frames_total / float(src.rate),
        getattr(src, "dropped_frames", 0), getattr(src, "overruns", 0)))
    print(json.dumps({"pipelines": [
        {"name": s.name, "audio_ms": s.total_ms, "dropped_16k_frames": s.dropped_16k,
         "segments": len(s.segments), "final": sum(1 for x in s.segments if x.is_final),
         "partial": sum(1 for x in s.segments if not x.is_final)} for s in segs]},
        ensure_ascii=False, indent=2))

    if args.out:
        for s in segs:
            sub = os.path.join(args.out, s.name.replace(":", "_")) if len(segs) > 1 else args.out
            os.makedirs(sub, exist_ok=True)
            for seg in s.segments:
                name = "seg%04d_utt%d_%d-%d_%s%s.wav" % (
                    seg.seg_id, seg.utterance_id, seg.start_ms, seg.end_ms,
                    seg.close_reason.value, "" if seg.is_final else "_partial")
                write_wav(os.path.join(sub, name), seg.samples, cfg.sample_rate)
            print("段文件已写入 %s（%d 个）" % (sub, len(s.segments)))
    return 0


def _make_source(args: argparse.Namespace):
    if args.input == "sim":
        # sim/wav 是非实时源：--duration 表示「合成多少秒音频」，而不是墙钟上限
        return SimSource(args.channels if args.channels != "max" else 2,
                         duration_s=args.duration if args.duration > 0 else 14.0,
                         rate=args.rate or 48000,
                         blocksize=args.blocksize or 480)
    if args.input == "wav":
        if not args.wav:
            raise ValueError("--input wav 需要给出 wav 路径")
        return WavSource(args.wav, blocksize=args.blocksize or 0)
    return MicSource(args)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时麦克风 VAD 分段 demo")
    p.add_argument("wav", nargs="?", default="", help="--input wav 时的文件路径")
    p.add_argument("--input", choices=("mic", "wav", "sim"), default="mic",
                   help="mic=麦克风（默认） wav=用文件模拟实时流 sim=合成多通道流")
    p.add_argument("--list-devices", action="store_true", help="列出输入设备后退出")
    p.add_argument("--input-device", default="", help="设备序号或名称子串，空=系统默认")
    p.add_argument("--channels", default="1", help="采集通道数，或 'max'（阵列全开）")
    p.add_argument("--rate", type=int, default=0, help="采集采样率，0=设备默认")
    p.add_argument("--blocksize", type=int, default=0, help="设备回调块大小（帧），0=10ms")
    p.add_argument("--mix", choices=("mean", "energy", "multi"), default="mean",
                   help="多通道收敛方式；multi=每通道独立状态机")
    p.add_argument("--channel", type=int, default=-1, help="固定取某一路（覆盖 --mix）")
    p.add_argument("--keep-db", type=float, default=3.0, help="energy 选通道的切换滞回")
    p.add_argument("--fade-blocks", type=int, default=2, help="切换时交叉淡化块数")
    p.add_argument("--qsize", type=int, default=50, help="采集队列长度（块数）")
    p.add_argument("--duration", type=float, default=0, help="采集多少秒后自动停止，0=不限")
    p.add_argument("--out", default="", help="段 wav 输出目录")
    # ---- 状态机 / VAD 参数（与 demo.py 一致）----
    p.add_argument("--sample-rate", type=int, default=16000, help="判决采样率（模型要求）")
    p.add_argument("--block-ms", type=int, default=32)
    p.add_argument("--ring-ms", type=int, default=8192)
    p.add_argument("--vad-lag", type=int, default=500)
    p.add_argument("--min-speech", type=int, default=300)
    p.add_argument("--min-silence", type=int, default=300)
    p.add_argument("--long-silence", type=int, default=650)
    p.add_argument("--max-utterance", type=int, default=5000)
    p.add_argument("--pre-roll", type=int, default=200)
    p.add_argument("--post-roll", type=int, default=100)
    p.add_argument("--prob-start", type=float, default=0.55)
    p.add_argument("--prob-end", type=float, default=0.35)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--emit-partial", action="store_true")
    p.add_argument("--vad", choices=("energy", "sherpa", "fsmn"), default="energy")
    p.add_argument("--threshold-db", type=float, default=-45.0)
    p.add_argument("--silero-model", default="", help="--vad sherpa 的 onnx 路径")
    p.add_argument("--funasr-model", default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    p.add_argument("--window-ms", type=int, default=1984)
    p.add_argument("--stride-ms", type=int, default=512)
    p.add_argument("--torch-device", default="cpu")
    p.add_argument("--width", type=int, default=56, help="时间轴宽度（字符）")
    p.add_argument("--no-heartbeat", action="store_true")
    p.add_argument("--quiet", action="store_true", help="不打印实时事件与报告，只留摘要")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _print_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError as exc:
        print("错误：需要 sounddevice（pip install sounddevice）：%s" % exc, file=sys.stderr)
        return 2
    try:
        devs = list(sd.query_devices())
    except Exception as exc:
        print("无法枚举音频设备：%s（这台机器大概率没有 PortAudio 设备）" % exc,
              file=sys.stderr)
        return 2
    print("%-4s %-42s %6s %6s %10s" % ("idx", "name", "in", "out", "rate"))
    for i, d in enumerate(devs):
        print("%-4d %-42s %6d %6d %10.0f%s" % (
            i, str(d["name"])[:42], d["max_input_channels"], d["max_output_channels"],
            d["default_samplerate"],
            "   <= 默认输入" if i == sd.default.device[0] else ""))
    ins = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    print("\n可用输入设备：%s" % (ins or "无（本机无麦克风，请用 --input sim / --input wav）"))
    return 0


def main(argv: List[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if args.list_devices:
        return _print_devices()
    try:
        return run(args)
    except ImportError as exc:
        print("错误：缺少依赖（%s）。麦克风采集需要 sounddevice；"
              "--vad sherpa 需要 sherpa-onnx；--vad fsmn 需要 funasr。"
              "无麦克风环境可用 --input sim 自测。" % exc, file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已手动停止", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
