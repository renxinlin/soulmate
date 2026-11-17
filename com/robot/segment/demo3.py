# -*- coding: utf-8 -*-
"""demo3：把状态机派发的**分段**拼成**整句**，导出完整音频 + 逐帧细节数组。

和 demo2 的分工
--------------
采集、重采样、下混、状态机、心跳**全部复用** demo2 的四个可复用接缝
（``make_config`` / ``_make_source`` / ``make_pipelines(disp_factory=…)`` / ``pump``），
本文件只做两件 demo2 不该管的事：

1. 用 :class:`UtteranceAssembler` **顶替**默认出口。demo2 的 ``Segmenter`` 早就为这一步
   留了 ``dispatcher`` 参数，所以状态机和环形缓冲零改动（文档 §5.6 的出口抽象就是为它留的）。
2. 起一条**消费线程**，把整句交给若干 sink：控制台可读视图、逐帧细节数组、
   wav + ``manifest.jsonl`` 落盘。

为什么必须在 engine 外面拼
--------------------------
``Segment.samples`` 只是**一段**。一句超过 ``max_utterance_ms`` 的话会被硬截成
``utt#N:0``、``utt#N:1``…（文档 §5.5 / R5），ASR 要的是整句，于是：

* 整句 = 同 ``utterance_id`` 的连续片之和，收尾标志是 ``close_reason != MAX_DURATION``；
* ``partial``（文档 §5.7）只是预览，与 final 同 ``seg_id``、区间更窄，**不入波形**；
* 续段起点不等于上段终点就是接缝异常：正空洞补零（保住「数组下标 ↔ 绝对毫秒」的映射）
  并记账，重叠裁头（绝不把同一段音频喂两遍）。

「语音细节数组」从哪来
----------------------
不在引擎里另开一份历史缓冲（``SegmentStats`` 只留 ``window_ms``≈2s 的滑窗，根本存不下
5s+ 的整句），而是拿**已经落到整句的 PCM** 按 ``block_ms`` 栅格重算电平，再用状态机导出的
``speech_start_ms`` / ``last_speech_ms``（判决时用的同一组边界）标出每帧是否属于语音、
是否落在补零空洞里。于是 :meth:`Utterance.frame_detail` 给出的就是与状态机同栅格的
逐帧数组，``index`` 能直接对回是哪一帧把句子切开。

用法::

    python3 -m com.robot.segment.demo3 --input sim --duration 14 --utt-out /tmp/utt3
    python3 -m com.robot.segment.demo3 --input wav /tmp/demo_in.wav --frames --arrays
    python3 -m com.robot.segment.demo3 --input sim --channels 2 --mix multi  # 每路一个装配器
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # 直接以脚本方式运行时补 sys.path
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from com.robot.segment import demo2
from com.robot.segment.collector import (
    BroadcastDispatcher, Utterance, UtteranceAssembler, UtteranceWriter)
from com.robot.segment.config import SegmentConfig
from com.robot.segment.dispatcher import CallbackDispatcher, ISegmentDispatcher

logger = logging.getLogger(__name__)

#: 电平条：rms_db 线性映射到 [-floor_db, 0] 的 8 档
_SPARK = "▁▂▃▄▅▆▇█"
SinkFn = Callable[[Utterance], None]


# ======================================================================
# 出口抽象：一个 sink 只拿整句、只读、且**绝不在采集线程里跑**
# ======================================================================
class IUtteranceSink(ABC):
    """整句出口的约定形态。

    实现者只需 :meth:`handle`；``__call__`` 已给出，因此
    :class:`~com.robot.segment.collector.UtteranceWriter` 这类「本来就是可调用对象」
    的出口可以直接塞进 :class:`UtterancePipeline`，不必再包一层适配器。
    """

    @abstractmethod
    def handle(self, utt: Utterance) -> None:
        ...

    def __call__(self, utt: Utterance) -> None:
        self.handle(utt)


def _bars(vals: Sequence[float], floor_db: float, width: int) -> str:
    """把电平序列画成定宽字符条（超过 width 则等间隔抽样，只作趋势观察）。"""
    if not vals:
        return ""
    step = max(1, -(-len(vals) // max(1, width)))
    picked = vals[::step]
    out = []
    for v in picked:
        k = int((v - floor_db) / (0.0 - floor_db) * (len(_SPARK) - 1))
        out.append(_SPARK[min(len(_SPARK) - 1, max(0, k))])
    return "".join(out) + ("（抽样 1/%d，共 %d 帧）" % (step, len(vals)) if step > 1 else "")


def _short_list(vals: Sequence[float], n: int = 24) -> str:
    head = ", ".join("%.1f" % v for v in vals[:n])
    return "[%s%s]" % (head, ", …+%d" % (len(vals) - n) if len(vals) > n else "")


class UtteranceConsole(IUtteranceSink):
    """人读视图：整句摘要 + 电平轨迹 + 逐帧细节（数组/表格按需）。"""

    def __init__(self, frame_ms: int = 32, floor_db: float = -60.0,
                 arrays: bool = False, rows: int = 0, width: int = 56) -> None:
        self.frame_ms = frame_ms
        self.floor_db = floor_db
        self.arrays = arrays
        self.rows = rows
        self.width = width
        self.count = 0

    def handle(self, utt: Utterance) -> None:
        self.count += 1
        frames = utt.frame_detail(self.frame_ms)
        rms = [f.rms_db for f in frames]
        peak = max((f.peak_db for f in frames), default=-120.0)
        voice = [f.rms_db for f in frames if f.in_speech]
        lo, hi = utt.speech_span_ms()
        print("\n" + "─" * self.width)
        print(utt.summary())
        print("  波形 %d 样本 / %dms @%dHz，帧 %d × %dms，峰值 %.1fdB，语音段均值 %.1fdB" % (
            utt.n_samples, utt.duration_ms, utt.sample_rate, len(frames), self.frame_ms,
            peak, sum(voice) / len(voice) if voice else -120.0))
        print("  语音区间 [%d,%d]（裁到这里的静音后再送 ASR）" % (lo, hi))
        print("  rms  %s" % _bars(rms, self.floor_db, self.width))
        if self.arrays:
            # 这三个数组就是「完整说话段的语音细节数组」的程序化出口
            print("  rms_db[]   = %s" % _short_list(rms))
            print("  peak_db[]  = %s" % _short_list([f.peak_db for f in frames]))
            print("  in_speech[]= %s" % "".join("1" if f.in_speech else "0" for f in frames)[:72]
                  + ("…" if len(frames) > 72 else ""))
            print("  zeroed[]   = %s" % "".join("1" if f.zeroed else "0" for f in frames)[:72]
                  + ("…" if len(frames) > 72 else ""))
        if self.rows > 0:
            print("  %5s %8s %8s %7s %7s %-7s %s" % (
                "idx", "start", "end", "rms", "peak", "speech", "zeroed"))
            for f in frames[:self.rows]:
                print("  %5d %8d %8d %7.1f %7.1f %-7s %s" % (
                    f.index, f.start_ms, f.end_ms, f.rms_db, f.peak_db,
                    f.in_speech, f.zeroed))
            if len(frames) > self.rows:
                print("  … 余 %d 帧（--rows 调大，或用 --frames 写进 manifest.jsonl）"
                      % (len(frames) - self.rows))
        for h in utt.holes:
            print("  ! 接缝 seg%d→seg%d：%s" % (
                h.from_seg, h.to_seg,
                "丢 %dms 已补零" % h.gap_ms if h.gap_ms > 0 else "重叠 %dms 已裁头" % -h.gap_ms))
        if utt.aborted:
            print("  ! 未等到收尾片（引擎复位/会话结束），aborted=True")
        sys.stdout.flush()


# ======================================================================
# 装配器 + 消费线程：采集线程只做拼接，落盘/打印都在自己的线程里
# ======================================================================
class UtterancePipeline:
    """一路流 = 一个 :class:`UtteranceAssembler` + 一条消费线程 + N 个 sink。

    采集线程里 ``dispatch()`` 只把 PCM 接进缓冲区（微秒级），真正的耗时——写 wav、
    算逐帧数组、塞给 ASR——发生在 :meth:`start` 起的线程内。队列由装配器的
    ``max_queued`` 兜底：消费不过来就丢最旧的整句并计数，**宁丢一句也不要无界涨内存**。

    多路采集（``--mix multi``）必须每路一个实例：``utterance_id`` 只在单路内单调。
    """

    def __init__(self, tag: str, assembler: UtteranceAssembler,
                 sinks: Sequence[SinkFn], poll_s: float = 0.2) -> None:
        self.tag = tag
        self.assembler = assembler
        self.sinks = list(sinks)
        self.poll_s = poll_s
        self.errors = 0
        self.forced = 0
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._th is not None:
            return
        self._th = threading.Thread(target=self._loop, name="utt-" + self.tag, daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while True:
            utt = self.assembler.take(self.poll_s)
            if utt is None:
                # 队列空 + 已要求停止才退出：停止前发布的整句一定已被取走
                if self._stop.is_set():
                    return
                continue
            self._emit(utt)

    def _emit(self, utt: Utterance) -> None:
        for s in self.sinks:
            try:
                s(utt)
            except Exception:                      # 单个 sink 出错不能拖垮整条消费线程
                self.errors += 1
                logger.exception("sink %s 处理 utt#%d 失败",
                                 type(s).__name__, utt.utterance_id)

    def stop(self, timeout: float = 5.0) -> int:
        """停止消费线程，并把「在途没等到收尾片」的整句强制交付。返回强制交付条数。

        先 join 再 :meth:`UtteranceAssembler.close_pending`，因此 sink 只会在一个线程里
        被调用；代价是 sink 卡死时 join 会超时，这里只告警不硬杀（Python 杀不掉线程）。
        """
        self._stop.set()
        th, self._th = self._th, None
        if th is not None:
            th.join(timeout)
            if th.is_alive():
                logger.warning("消费线程 %s 仍在跑（sink 卡住？），停止交付", th.name)
        for utt in self.assembler.close_pending():
            self._emit(utt)
            self.forced += 1
        return self.forced

    def stats(self) -> Dict[str, object]:
        st = dict(self.assembler.stats())
        st["sink_errors"] = self.errors
        st["forced_delivered"] = self.forced
        return st


# ======================================================================
# 组装：demo2 的管线 + 本文件的装配层
# ======================================================================
def build_sinks(args: argparse.Namespace, frame_ms: int,
                writers: List[UtteranceWriter]) -> List[SinkFn]:
    sinks: List[SinkFn] = []
    if not args.quiet:
        sinks.append(UtteranceConsole(frame_ms=frame_ms, floor_db=args.floor_db,
                                      arrays=args.arrays, rows=args.rows, width=args.width))
    if args.utt_out:
        w = UtteranceWriter(args.utt_out, frames=args.frames, frame_ms=frame_ms,
                            pad_ms=args.trim_pad, keep_trimmed=not args.no_trim)
        writers.append(w)
        sinks.append(w)                            # UtteranceWriter 已是可调用对象
    return sinks


def make_assemblers(args: argparse.Namespace, cfg: SegmentConfig, src,
                    writers: List[UtteranceWriter]
                    ) -> Tuple[List[demo2.Segmenter], Optional[demo2.Downmix],
                               List[UtterancePipeline], int]:
    """建 demo2 管线，并把每路的出口换成整句装配器。

    返回 ``(segs, dm, pipelines, frame_ms)``。
    """
    frame_ms = args.frame_ms or cfg.block_ms
    pipes: List[UtterancePipeline] = []

    def factory(name: str) -> ISegmentDispatcher:
        tag = name.replace(":", "_")               # 目录名要文件系统安全（mix:energy → mix_energy）
        asm = UtteranceAssembler(tag=tag, max_queued=args.utt_queue)
        pipes.append(UtterancePipeline(name, asm, build_sinks(args, frame_ms, writers)))
        if args.show_segments:                     # 想同时看段级细节时广播一份
            cb = CallbackDispatcher(lambda seg, n=name: print("[seg:%s] %s" % (n, seg.summary()),
                                                              flush=True))
            return BroadcastDispatcher([asm, cb])
        return asm

    segs, dm = demo2.make_pipelines(args, cfg, src, disp_factory=factory)
    return segs, dm, pipes, frame_ms


def run(args: argparse.Namespace) -> int:
    cfg = demo2.make_config(args)
    src = demo2._make_source(args)
    writers: List[UtteranceWriter] = []
    with src:
        segs, dm, pipes, frame_ms = make_assemblers(args, cfg, src, writers)
        print("输入：%s @ %dHz / %d 通道 → %dHz，%s" % (
            src.device_name, src.rate, src.channels, cfg.sample_rate,
            ("每通道一套装配器（%d 路）" % len(segs)) if dm is None else ("下混：" + dm.mode)))
        print("整句出口：%s；逐帧栅格 %dms（=%s block_ms，index 与状态机帧序一致）" % (
            (args.utt_out or "仅控制台"), frame_ms,
            "SegmentConfig" if not args.frame_ms else "--frame-ms 自定义"))
        if args.frame_ms and args.frame_ms != cfg.block_ms:
            print("[note] --frame-ms 与 block_ms 不同，frame_detail 的 index 不再是状态机的帧号，"
                  "只作电平观察用")
        for p in pipes:
            p.start()
        stats = demo2.pump(src, segs, dm, args)

    for s in segs:                                 # §7-G15：先让引擎闭合在途段
        s.flush()
    for p in pipes:                                # 再收装配层，顺序反了会丢最后一句
        p.stop()

    total_ms = max([s.total_ms for s in segs] or [0])
    print("\n采集 %d 帧（%.2fs，音频时钟 %.2fs）丢弃=%d 次 墙钟=%.1fs" % (
        stats["frames"], stats["frames"] / float(src.rate), total_ms / 1000.0,
        getattr(src, "dropped_frames", 0), stats["wall_ms"] / 1000.0))
    for p in pipes:
        st = p.stats()
        print("  %-12s 片=%s 整句=%s 预览=%s 空洞=%s 队列丢=%s 强制交付=%s 出口错=%s" % (
            p.tag, st["parts"], st["finished"], st["previews"], st["holes"],
            st["dropped_queued"], st["forced_delivered"], st["sink_errors"]))
    if writers:
        n = sum(w.written for w in writers)
        print("整句已写入 %s：%d 条（整句 wav + 裁剪 wav + manifest.jsonl%s）" % (
            args.utt_out, n, "，含逐帧细节数组" if args.frames else ""))
    print(json.dumps({"pipelines": [p.stats() for p in pipes]}, ensure_ascii=False))
    return 0


# ======================================================================
# CLI：复用 demo2 的全部参数，只加整句相关项
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = demo2.build_parser()
    p.description = "VAD 分段 → 整句装配 demo（完整音频 + 逐帧细节数组）"
    g = p.add_argument_group("整句出口")
    g.add_argument("--utt-out", default="", help="整句输出目录（wav + manifest.jsonl），空=不落盘")
    g.add_argument("--utt-queue", type=int, default=8, help="待消费整句队列上限，满了丢最旧")
    g.add_argument("--frame-ms", type=int, default=0,
                   help="逐帧细节栅格（ms），0=用 block_ms 以保持与状态机帧序对齐")
    g.add_argument("--frames", action="store_true", help="把逐帧细节数组写进 manifest.jsonl")
    g.add_argument("--arrays", action="store_true", help="控制台打印 rms/peak/in_speech/zeroed 数组")
    g.add_argument("--rows", type=int, default=0, help="控制台打印前 N 帧明细表")
    g.add_argument("--floor-db", type=float, default=-60.0, help="电平条下限")
    g.add_argument("--trim-pad", type=int, default=40, help="裁剪整句时语音区间外保留的余量（ms）")
    g.add_argument("--no-trim", action="store_true", help="不写 *_trim.wav")
    g.add_argument("--show-segments", action="store_true", help="同时打印段级派发（广播一份给打印出口）")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if args.list_devices:
        return demo2._print_devices()
    try:
        return run(args)
    except ImportError as exc:
        print("错误：缺少依赖（%s）。麦克风采集需要 sounddevice；无麦克风环境用 --input sim 自测。"
              % exc, file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已手动停止（仍会闭合在途段并交付整句）", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
