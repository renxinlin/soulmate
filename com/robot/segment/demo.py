# -*- coding: utf-8 -*-
"""离线回放 demo：``wav → VAD → 分段状态机 → 段文件 + 时间轴报告``。

用来在没有麦克风、没有 ASR 服务的情况下验证设计文档 §8 的可观测指标：
一条长语音 wav 进去，能看到状态机把哪儿切成段、为什么切、切得对不对。

用法（仓库根目录，纯标准库即可跑）::

    python -m com.robot.segment.demo --vad energy long_16k_mono.wav
    python -m com.robot.segment.demo --vad energy --emit-partial --out /tmp/seg a.wav
    python -m com.robot.segment.demo --vad fsmn --funasr-model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch a.wav

输入必须是 16kHz / 单声道 / 16bit PCM wav，否则先转换::

    ffmpeg -i in.mp3 -ar 16000 -ac 1 -sample_fmt s16 out.wav

``--vad energy`` 用电能 VAD（无第三方依赖，适合先看分段逻辑本身）；
``--vad fsmn`` 走 FunASR 的 FSMN-VAD，经 :class:`WindowSegmentVad` 适配成逐帧判决，
与 ``com/robot/models/asr/asr.py`` 里那套「整窗重算」的用法等价。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import wave
from typing import List

if __package__ in (None, ""):  # 直接以脚本方式运行时补 sys.path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from com.robot.segment.config import SegmentConfig
from com.robot.segment.dispatcher import CallbackDispatcher
from com.robot.segment.engine import SegmentEngine
from com.robot.segment.models import Segment
from com.robot.segment.ring import AudioRingBuffer
from com.robot.segment.vad import EnergyVad, WindowSegmentVad


# ====================================================================== 输入
def read_pcm(path: str, sample_rate: int) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: 需要单声道 16bit PCM，实际 ch={w.getnchannels()} "
                             f"width={w.getsampwidth()}")
        if w.getframerate() != sample_rate:
            raise ValueError(f"{path}: 采样率 {w.getframerate()}Hz ≠ 配置 {sample_rate}Hz")
        return w.readframes(w.getnframes())


def write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


# ====================================================================== VAD
def build_vad(args: argparse.Namespace, cfg: SegmentConfig):
    if args.vad == "energy":
        return EnergyVad(threshold_db=args.threshold_db)
    from funasr import AutoModel  # 惰性导入：只有真要用 FSMN 才依赖 funasr

    model = AutoModel(model=args.funasr_model, device=args.device)
    vad = WindowSegmentVad(
        model,
        sample_rate=cfg.sample_rate,
        block_ms=cfg.block_ms,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
    )
    print(f"[vad] FSMN 批量判决：window={vad.window_ms}ms stride={vad.stride_ms}ms "
          f"固有滞后={vad.lag_ms}ms（配置里的 vad_lag={cfg.vad_lag_ms}ms）")
    if cfg.vad_lag_ms < vad.lag_ms:
        print(f"[warn] vad_lag_ms({cfg.vad_lag_ms}) < 批量判决滞后({vad.lag_ms}ms)，"
              f"状态机拿到的概率与它以为的帧时间不一致")
    return vad


# ====================================================================== 报告
def timeline(segments: List[Segment], total_ms: int, width: int = 56) -> str:
    """把派发区间画成 ASCII 时间轴，肉眼检查重叠 / 空洞 / pre_roll。"""
    scale = width / max(1, total_ms)
    lines = []
    for seg in segments:
        a = int(seg.start_ms * scale)
        b = max(a + 1, int(seg.end_ms * scale))
        row = [" "] * width
        row[a:b] = ["=" if seg.is_final else "~"] * (b - a)
        tag = seg.close_reason.value + ("" if seg.is_final else "/partial")
        lines.append("%6dms %5.2fs %s  %s" % (seg.start_ms, seg.start_ms / 1000.0,
                                              "".join(row).rstrip(), tag))
    return "\n".join(lines)


def report(segments: List[Segment], engine: SegmentEngine, ring: AudioRingBuffer,
           total_ms: int, width: int = 56) -> None:
    print("\n=== 段列表 ===")
    header = "%-5s %-8s %9s %9s %8s %-13s %-8s %6s %7s" % (
        "seg", "utt:seq", "start", "end", "dur", "close", "status", "conf", "decision")
    print(header)
    print("-" * len(header))
    for s in segments:
        print("%-5d %-8s %9d %9d %8d %-13s %-8s %6.2f %7d" % (
            s.seg_id, "%d:%d" % (s.utterance_id, s.seq_in_utterance), s.start_ms, s.end_ms,
            s.duration_ms, s.close_reason.value, s.status.value, s.vad_confidence, s.decision_ms))

    print("\n=== 时间轴（=/final ~=partial，总时长 %.1fs）===" % (total_ms / 1000.0))
    print(timeline(segments, total_ms, width=width))

    # 覆盖检查：相邻段之间既不该重叠也不该留空洞（同句拼接尤其如此）
    finals = [s for s in segments if s.is_final]
    overlaps = [(a.seg_id, b.seg_id) for a, b in zip(finals, finals[1:]) if b.start_ms < a.end_ms]
    holes = [(a.seg_id, b.seg_id, b.start_ms - a.end_ms)
             for a, b in zip(finals, finals[1:]) if b.start_ms > a.end_ms]
    print("\n=== 覆盖检查 ===")
    print("重叠段对: %s" % (overlaps or "无"))
    print("空洞段对: %s" % (holes or "无"))

    print("\n=== 游标与丢弃 ===")
    print("W=%dms C=%dms D=%dms ring_capacity=%dms dropped=%dms" % (
        engine.w_ms, engine.c_ms, engine.d_ms, ring.capacity_ms, ring.dropped_ms))

    print("\n=== 指标 ===")
    print(json.dumps(engine.stats.report(), ensure_ascii=False, indent=2, sort_keys=True))


# ====================================================================== 主流程
def run(args: argparse.Namespace) -> int:
    cfg = SegmentConfig(
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        ring_capacity_ms=args.ring_ms,
        vad_lag_ms=args.vad_lag,
        min_speech_ms=args.min_speech,
        min_silence_ms=args.min_silence,
        max_utterance_ms=args.max_utterance,
        pre_roll_ms=args.pre_roll,
        post_roll_ms=args.post_roll,
        prob_start=args.prob_start,
        prob_end=args.prob_end,
        emit_partial=args.emit_partial,
        warmup_ms=args.warmup,
    )
    cfg.validate()

    pcm = read_pcm(args.wav, cfg.sample_rate)
    total_ms = len(pcm) // cfg.bytes_per_ms
    vad = build_vad(args, cfg)

    ring = AudioRingBuffer(cfg)
    segments: List[Segment] = []
    engine = SegmentEngine(cfg, ring, CallbackDispatcher(segments.append))

    print("输入 %s：%dms @%dHz/%dB" % (os.path.basename(args.wav), total_ms,
                                       cfg.sample_rate, cfg.channels))
    print("参数：block=%dms vad_lag=%dms min_speech=%dms min_silence=%dms max_utter=%dms "
          "pre=%dms post=%dms emit_partial=%s" % (
              cfg.block_ms, cfg.vad_lag_ms, cfg.min_speech_ms, cfg.min_silence_ms,
              cfg.max_utterance_ms, cfg.pre_roll_ms, cfg.post_roll_ms, cfg.emit_partial))

    step = cfg.block_samples * 2
    t = 0
    for off in range(0, len(pcm) - len(pcm) % step, step):
        chunk = pcm[off:off + step]
        ring.write(chunk, t)
        engine.feed(chunk, t, vad)
        t += cfg.block_ms

    engine.flush()                        # §7-G15：采集停止必须闭合在途段（经派发器入列）

    report(segments, engine, ring, total_ms, width=args.width)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for s in segments:
            name = "seg%04d_utt%d_%d-%d_%s%s.wav" % (
                s.seg_id, s.utterance_id, s.start_ms, s.end_ms, s.close_reason.value,
                "" if s.is_final else "_partial")
            path = os.path.join(args.out, name)
            write_wav(path, s.samples, cfg.sample_rate)
        print("\n段文件已写入 %s（%d 个）" % (args.out, len(segments)))
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAD 分段状态机离线回放")
    p.add_argument("wav", help="16kHz/mono/16bit PCM wav")
    p.add_argument("--vad", choices=("energy", "fsmn"), default="energy")
    p.add_argument("--out", default="", help="段 wav 输出目录，留空则不写文件")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--block-ms", type=int, default=32)
    p.add_argument("--ring-ms", type=int, default=8192)
    p.add_argument("--vad-lag", type=int, default=500)
    p.add_argument("--min-speech", type=int, default=300)
    p.add_argument("--min-silence", type=int, default=300)
    p.add_argument("--max-utterance", type=int, default=5000)
    p.add_argument("--pre-roll", type=int, default=200)
    p.add_argument("--post-roll", type=int, default=100)
    p.add_argument("--prob-start", type=float, default=0.55)
    p.add_argument("--prob-end", type=float, default=0.35)
    p.add_argument("--threshold-db", type=float, default=-45.0, help="energy VAD 门限")
    p.add_argument("--warmup", type=int, default=0, help="丢弃上电 pop 音的毫秒数")
    p.add_argument("--emit-partial", action="store_true", help="流式提前送（§5.7）")
    p.add_argument("--window-ms", type=int, default=1984, help="fsmn：整窗重算长度")
    p.add_argument("--stride-ms", type=int, default=512, help="fsmn：重算步进")
    p.add_argument("--funasr-model", default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    p.add_argument("--device", default="cpu")
    p.add_argument("--width", type=int, default=56, help="时间轴宽度（字符）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        return run(args)
    except ImportError as exc:
        print("错误：--vad fsmn 需要 funasr，当前环境不可用（%s）。改用 --vad energy 即可。" % exc,
              file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
