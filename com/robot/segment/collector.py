# -*- coding: utf-8 -*-
"""整句装配层：把状态机派发的**分段**还原成**一句话的完整音频 + 逐帧细节数组**。

为什么需要这一层
----------------
:meth:`SegmentEngine.step` 每闭合一段就派发一个 :class:`~com.robot.segment.models.Segment`，
``Segment.samples`` 已经是该段的完整 PCM（int16 LE，从环形缓冲深拷贝而来，
脱离缓冲后可安全持有）。但一句长话会被 ``max_utterance_ms`` 硬截成
``utt#N:0``、``utt#N:1``… 若干片（文档 §5.5 / R5），ASR 要的是**整句**，
所以需要一个装配层按 ``utterance_id`` 拼接。拼接时同时校验文档 §5.5 的三条约定：

* 续段起点必须 ``==`` 上一段终点。不等就是**空洞**（正=丢音频，负=重叠），
  正空洞补零以保住「数组下标 ↔ 绝对毫秒」的映射，并记进 :attr:`Utterance.holes`；
  负载重叠则裁掉重复的头部，绝不把同一段音频喂两遍。
  注意：真正的采集丢帧在**引擎层**就已被 E7 强制闭合（段带 ``status=gap``，
  下一片换 ``utterance_id``），所以正常运行时这里不该出现正空洞——出现即说明
  §5.5 的落库约定被破坏（例如 ``clamped``），要查上游，补零只是兜底而非掩盖。
* ``partial`` 段（文档 §5.7）只是预览，与 final 同 ``seg_id``、区间更窄，
  **不参与拼接**，只挂到 :attr:`Utterance.preview`。
* 收尾判定：``close_reason != MAX_DURATION`` 的那一片才算这句说完。

三个数据出口
------------
1. :meth:`Utterance.pcm` / :meth:`Utterance.to_int16` / :meth:`Utterance.to_numpy`
   —— 整句波形数组；
2. :meth:`Utterance.frame_detail` —— 与 ``block_ms`` 同栅格的**逐帧细节数组**
   （每帧 rms/peak 电平、是否落在 VAD 语音区间、是否落在补零空洞）；
3. :meth:`Utterance.trim_to_speech` —— 按 VAD 的 ``speech_start_ms`` /
   ``last_speech_ms`` 裁掉首尾静音，这才是真正该送 ASR 的那一段。

线程模型
--------
:meth:`UtteranceAssembler.dispatch` 由**采集/VAD 线程**同步调用，锁内只做拼接，
不碰文件：整句经 :meth:`UtteranceAssembler.take` 交给消费线程（ASR / 落盘）。
``on_complete`` 若设置，会在采集线程内同步执行——只适合打日志这类轻活。
"""
from __future__ import annotations

import array
import json
import math
import os
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .dispatcher import ISegmentDispatcher
from .models import CloseReason, Segment

#: 只有硬截断片需要续接；其余闭合原因都表示「这句说完了」
_OPEN_REASON = CloseReason.MAX_DURATION


# ======================================================================
# 数据结构
# ======================================================================
@dataclass(frozen=True)
class Hole:
    """同句两片之间的接缝异常。``gap_ms > 0`` 丢音频（已补零），``< 0`` 重叠（已裁头）。"""

    from_seg: int
    to_seg: int
    gap_ms: int


@dataclass(frozen=True)
class FrameDetail:
    """一个判决帧的音频细节，栅格与 ``SegmentConfig.block_ms`` 一致。"""

    index: int
    start_ms: int                  # 相对整句起点
    end_ms: int
    rms_db: float
    peak_db: float
    in_speech: bool                # 是否落在 VAD 语音区间内
    zeroed: bool                   # 是否落在补零空洞内（这段音频不是真的）

    def as_dict(self) -> Dict[str, object]:
        return {"i": self.index, "start_ms": self.start_ms, "end_ms": self.end_ms,
                "rms_db": round(self.rms_db, 1), "peak_db": round(self.peak_db, 1),
                "in_speech": self.in_speech, "zeroed": self.zeroed}


def _levels(samples: Sequence[int]) -> Tuple[float, float]:
    """返回 ``(rms_dbfs, peak_dbfs)``；静音返回 -120。满量程按 int16 的 32768。"""
    if not samples:
        return -120.0, -120.0
    acc = 0
    peak = 0
    for v in samples:
        acc += v * v
        a = -v if v < 0 else v
        if a > peak:
            peak = a
    if peak <= 0:
        return -120.0, -120.0
    rms = math.sqrt(acc / len(samples))
    return (20.0 * math.log10(rms / 32768.0) if rms > 1.0 else -120.0,
            20.0 * math.log10(peak / 32768.0))


# ======================================================================
# 整句
# ======================================================================
class Utterance:
    """一句话（可能由多个硬截断片拼成）的音频与元数据。"""

    def __init__(self, utterance_id: int, sample_rate: int, tag: str = "") -> None:
        self.utterance_id = utterance_id
        self.sample_rate = sample_rate
        self.tag = tag
        self.parts: List[Segment] = []
        self.holes: List[Hole] = []
        self.preview: Optional[Segment] = None       # 最近一条 partial（文档 §5.7）
        self.complete = False
        self.close_reason: Optional[str] = None
        self.aborted = False                         # 没等到收尾片就被新句/复位顶掉
        self._pcm = bytearray()
        self._zero_spans: List[Tuple[int, int]] = []  # 补零区间的绝对毫秒范围
        self._frame_cache: Dict[int, List[FrameDetail]] = {}

    # ------------------------------------------------------------------ 累积
    def append(self, seg: Segment) -> None:
        """并入一片 final 音频。partial 只更新 :attr:`preview`，不进波形。"""
        if seg.partial:
            self.preview = seg
            return
        if seg.sample_rate != self.sample_rate:
            raise ValueError("同一句里采样率必须一致：%d vs %d" % (self.sample_rate, seg.sample_rate))

        if self.parts:
            prev = self.parts[-1]
            if seg.seq_in_utterance != prev.seq_in_utterance + 1:
                raise ValueError("seq 不连续：%d → %d（装配假设同一 utterance_id 内逐片递增）"
                                 % (prev.seq_in_utterance, seg.seq_in_utterance))
            if seg.start_ms > prev.end_ms:                       # 空洞 → 补零
                gap = seg.start_ms - prev.end_ms
                self.holes.append(Hole(prev.seg_id, seg.seg_id, gap))
                pad = gap * self.sample_rate // 1000 * 2
                self._pcm += b"\x00" * pad
                self._zero_spans.append((prev.end_ms, seg.start_ms))
            elif seg.start_ms < prev.end_ms:                      # 重叠 → 裁掉重复头部
                over = prev.end_ms - seg.start_ms
                self.holes.append(Hole(prev.seg_id, seg.seg_id, -over))
                head = min(len(seg.samples), over * self.sample_rate // 1000 * 2)
                seg = _without_head(seg, head)
        self._pcm += seg.samples
        self.parts.append(seg)
        if seg.close_reason is not _OPEN_REASON:
            self.complete = True
            self.close_reason = seg.close_reason.value
        self._frame_cache.clear()

    # ------------------------------------------------------------------ 只读属性
    @property
    def pcm(self) -> bytes:
        """整句 PCM（int16 LE、单声道），含补零空洞。"""
        return bytes(self._pcm)

    @property
    def n_samples(self) -> int:
        return len(self._pcm) // 2

    @property
    def start_ms(self) -> int:
        return self.parts[0].start_ms if self.parts else 0

    @property
    def end_ms(self) -> int:
        return self.parts[-1].end_ms if self.parts else 0

    @property
    def duration_ms(self) -> int:
        """波形时长（= 音频真实长度，含补零）。"""
        return self.n_samples * 1000 // self.sample_rate

    @property
    def contiguous(self) -> bool:
        return not self.holes

    @property
    def hole_ms(self) -> int:
        return sum(abs(h.gap_ms) for h in self.holes)

    # ------------------------------------------------------------------ 数组出口
    def to_int16(self) -> array.array:
        """整句样本数组 ``array('h')``（纯标准库，可直接下标/切片）。"""
        a = array.array("h")
        a.frombytes(self.pcm)
        return a

    def to_numpy(self):
        """整句 ``numpy.int16`` 一维数组（惰性导入 numpy）。"""
        import numpy as np

        return np.frombuffer(self.pcm, dtype=np.int16)

    def ms_to_index(self, abs_ms: int) -> int:
        """绝对毫秒 → 本句数组下标。"""
        idx = (abs_ms - self.start_ms) * self.sample_rate // 1000
        return max(0, min(self.n_samples, idx))

    def index_to_ms(self, idx: int) -> int:
        """本句数组下标 → 绝对毫秒。"""
        return self.start_ms + idx * 1000 // self.sample_rate

    # ------------------------------------------------------------------ VAD 语音区间
    def speech_spans_ms(self) -> List[Tuple[int, int]]:
        """各片的真实语音起止（绝对毫秒）。

        用的是状态机导出的 ``speech_start_ms`` / ``last_speech_ms``，
        比在波形上重新做能量判决可信——那正是判决时用的同一组边界。
        """
        return [(s.speech_start_ms, s.last_speech_ms) for s in self.parts
                if s.last_speech_ms > s.speech_start_ms]

    def speech_span_ms(self) -> Tuple[int, int]:
        spans = self.speech_spans_ms()
        return (spans[0][0], spans[-1][1]) if spans else (self.start_ms, self.end_ms)

    def trim_to_speech(self, pad_ms: int = 40, fallback_energy_db: float = -45.0) -> array.array:
        """裁掉首尾静音，返回真正该送 ASR 的样本数组。

        边界优先用 VAD 区间；缺失（例如外部构造的句子）时退回能量法。
        ``pad_ms`` 是给辅音头尾留的余量，别用 0。
        """
        a = self.to_int16()
        if not len(a):
            return a
        lo_ms, hi_ms = self.speech_span_ms()
        if hi_ms <= lo_ms:
            hot = [f for f in self.frame_detail(32) if f.rms_db >= fallback_energy_db]
            if not hot:
                return a
            lo_ms = self.start_ms + hot[0].start_ms
            hi_ms = self.start_ms + hot[-1].end_ms
        i0 = self.ms_to_index(lo_ms - pad_ms)
        i1 = self.ms_to_index(hi_ms + pad_ms)
        return a[i0:i1] if i1 > i0 else a

    # ------------------------------------------------------------------ 逐帧细节
    def frame_detail(self, frame_ms: int = 32) -> List[FrameDetail]:
        """逐帧细节数组：``(rms_db, peak_db, in_speech, zeroed)``。

        ``frame_ms`` 传 ``SegmentConfig.block_ms`` 时，``index`` 与状态机的帧序一致，
        可以直接对回是哪一帧把句子切开的。结果按 ``frame_ms`` 缓存。
        """
        cached = self._frame_cache.get(frame_ms)
        if cached is not None:
            return cached
        sps = max(1, self.sample_rate * frame_ms // 1000)
        a = self.to_int16()
        speech = self.speech_spans_ms()
        out: List[FrameDetail] = []
        for i in range((len(a) + sps - 1) // sps):
            blk = a[i * sps:(i + 1) * sps]
            if not blk:
                break
            lo = self.start_ms + i * frame_ms
            hi = lo + len(blk) * 1000 // self.sample_rate
            rms, peak = _levels(blk)
            out.append(FrameDetail(
                index=i, start_ms=i * frame_ms,
                end_ms=len(blk) * 1000 // self.sample_rate + i * frame_ms,
                rms_db=rms, peak_db=peak,
                in_speech=any(s < hi and e > lo for s, e in speech),
                zeroed=any(x < hi and y > lo for x, y in self._zero_spans)))
        self._frame_cache[frame_ms] = out
        return out

    # ------------------------------------------------------------------ 出口
    def manifest(self, with_frames: bool = False, frame_ms: int = 32) -> Dict[str, object]:
        """JSON 可序列化的整句描述；``with_frames=True`` 时附完整逐帧细节数组。"""
        lo, hi = self.speech_span_ms()
        out: Dict[str, object] = {
            "tag": self.tag,
            "utterance_id": self.utterance_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "n_samples": self.n_samples,
            "sample_rate": self.sample_rate,
            "complete": self.complete,
            "aborted": self.aborted,
            "close_reason": self.close_reason,
            "contiguous": self.contiguous,
            "hole_ms": self.hole_ms,
            "holes": [{"from_seg": h.from_seg, "to_seg": h.to_seg, "gap_ms": h.gap_ms}
                      for h in self.holes],
            "speech_span_ms": [lo, hi],
            "speech_spans_ms": [list(x) for x in self.speech_spans_ms()],
            "has_preview": self.preview is not None,
            "parts": [{
                "seg_id": s.seg_id, "seq": s.seq_in_utterance,
                "start_ms": s.start_ms, "end_ms": s.end_ms,
                "duration_ms": s.duration_ms, "close_reason": s.close_reason.value,
                "status": s.status.value, "vad_confidence": s.vad_confidence,
                "decision_ms": s.decision_ms,
                "speech_start_ms": s.speech_start_ms, "last_speech_ms": s.last_speech_ms,
                "noise_floor_db": s.noise_floor_db,
                "mean_level_db": s.meta.get("mean_level_db"),
                "peak_level_db": s.meta.get("peak_level_db"),
                "peak_prob": s.meta.get("peak_prob"),
                "needs_stitch": s.meta.get("needs_stitch"),
            } for s in self.parts],
        }
        if with_frames:
            out["frames"] = [f.as_dict() for f in self.frame_detail(frame_ms)]
        return out

    def summary(self) -> str:
        lo, hi = self.speech_span_ms()
        return ("utt#%d%s %d片 [%d,%d] %dms %s%s 语音[%d,%d] %s" % (
            self.utterance_id, ("@%s" % self.tag) if self.tag else "",
            len(self.parts), self.start_ms, self.end_ms, self.duration_ms,
            self.close_reason or "open", "/aborted" if self.aborted else "",
            lo, hi, ("空洞%dms" % self.hole_ms) if self.holes else "连续"))

    def __repr__(self) -> str:  # pragma: no cover - 只用于日志
        return "Utterance(%s, %dB)" % (self.summary(), len(self._pcm))


def _without_head(seg: Segment, drop_bytes: int) -> Segment:
    """重叠裁剪：返回去掉前 ``drop_bytes`` 的新 Segment，绝对时间起点同步后移。"""
    drop_bytes -= drop_bytes % 2
    return Segment(
        seg_id=seg.seg_id, utterance_id=seg.utterance_id, seq_in_utterance=seg.seq_in_utterance,
        start_ms=seg.start_ms + drop_bytes * 1000 // (seg.sample_rate * 2),
        end_ms=seg.end_ms, duration_ms=seg.duration_ms - drop_bytes * 1000 // (seg.sample_rate * 2),
        samples=seg.samples[drop_bytes:],
        sample_rate=seg.sample_rate, channels=seg.channels, close_reason=seg.close_reason,
        status=seg.status, vad_confidence=seg.vad_confidence, noise_floor_db=seg.noise_floor_db,
        partial=seg.partial, decision_ms=seg.decision_ms, speech_start_ms=seg.speech_start_ms,
        last_speech_ms=seg.last_speech_ms, meta=dict(seg.meta, head_trimmed_bytes=drop_bytes))


# ======================================================================
# 装配器：状态机出口 → 整句队列
# ======================================================================
class UtteranceAssembler(ISegmentDispatcher):
    """把 :class:`Segment` 流装配成 :class:`Utterance` 流。

    插在 :class:`~com.robot.segment.engine.SegmentEngine` 的 dispatcher 位置上即可，
    状态机与环形缓冲**不需要任何改动**（文档 §5.6 的出口抽象就是为这一步留的）。

    Parameters
    ----------
    tag           : 流名。多路采集（``--mix multi``）时**每路一个装配器**——
                    ``utterance_id`` 只在单路内单调，跨路不可比。
    on_complete   : 整句收尾时在采集线程同步回调，只适合打日志/入队等轻活。
    keep_finished : 额外留存最近 N 条整句供 :meth:`finished` 查询，默认 0（不留，省内存）。
    max_queued    : :meth:`take` 队列上限，满了丢最旧并计数；宁丢一句也不要无界涨内存。
    """

    def __init__(self, tag: str = "",
                 on_complete: Optional[Callable[[Utterance], None]] = None,
                 keep_finished: int = 0, max_queued: int = 8) -> None:
        self.tag = tag
        self.on_complete = on_complete
        self.max_queued = max(1, max_queued)
        self._cv = threading.Condition()
        self._open: Optional[Utterance] = None
        self._done: Deque[Utterance] = deque()
        self._keep: Deque[Utterance] = deque(maxlen=max(0, keep_finished))
        self.parts_seen = 0
        self.previews_seen = 0
        self.finished = 0
        self.holes_seen = 0
        self.dropped_queued = 0

    # ------------------------------------------------------------------ 生产侧（采集线程）
    def dispatch(self, seg: Segment) -> bool:
        if seg.partial:
            with self._cv:
                self.previews_seen += 1
                if self._open is not None and self._open.utterance_id == seg.utterance_id:
                    self._open.preview = seg
            return True

        ready: Optional[Utterance] = None
        with self._cv:
            cur = self._open
            if cur is None or seg.utterance_id != cur.utterance_id:
                if cur is not None:
                    # 没等到收尾片就来了新句：多半是引擎 reset / 时间轴重启
                    self._publish(cur, aborted=True)
                cur = self._open = Utterance(seg.utterance_id, seg.sample_rate, self.tag)
            before = len(cur.holes)
            cur.append(seg)
            self.holes_seen += len(cur.holes) - before
            self.parts_seen += 1
            if cur.complete:
                self._open = None
                ready = cur
                self._publish(cur)
            if ready is not None:
                self._cv.notify_all()
        if ready is not None and self.on_complete is not None:
            self.on_complete(ready)          # 锁外回调：允许消费者 take() 不互相阻塞
        return True

    # ------------------------------------------------------------------ 消费侧（ASR / 落盘线程）
    def take(self, timeout: Optional[float] = None) -> Optional[Utterance]:
        """取一条已完成整句。``timeout=None`` 非阻塞；给数值则最多等这么久。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while not self._done:
                if deadline is None:
                    return None
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return self._done.popleft()

    def drain(self) -> List[Utterance]:
        with self._cv:
            out = list(self._done)
            self._done.clear()
        return out

    def current(self) -> Optional[Utterance]:
        """在途未收尾的整句（只读语义），流式想看长度/片数时用。"""
        with self._cv:
            return self._open

    def finished(self) -> List[Utterance]:
        """``keep_finished > 0`` 时可按序号回查的整句。"""
        with self._cv:
            return list(self._keep)

    def close_pending(self, reason: str = "session_end") -> List[Utterance]:
        """会话结束：把在途整句强制收尾（``aborted=True``）并返回它们。

        正常路径下 :meth:`SegmentEngine.flush` 已经用 ``EOS`` 把最后一段派发出来，
        所以这里只会兜到「引擎中途停用」这类异常，返回空列表才是常态。
        """
        out: List[Utterance] = []
        with self._cv:
            if self._open is not None:
                utt, self._open = self._open, None
                utt.close_reason = utt.close_reason or reason
                self._publish(utt, aborted=True)
                self._cv.notify_all()
            out = list(self._done)
            self._done.clear()
        return out

    def stats(self) -> Dict[str, object]:
        with self._cv:
            return {"tag": self.tag or None, "parts": self.parts_seen,
                    "previews": self.previews_seen, "finished": self.finished,
                    "holes": self.holes_seen, "queued": len(self._done),
                    "dropped_queued": self.dropped_queued,
                    "open": None if self._open is None else self._open.utterance_id}

    def __repr__(self) -> str:  # pragma: no cover
        return "UtteranceAssembler(%s)" % (self.stats(),)

    # ------------------------------------------------------------------ 内部
    def _publish(self, utt: Utterance, aborted: bool = False) -> None:
        """调用方必须持锁：标记完成、入队、必要时挤掉最旧的。"""
        utt.aborted = aborted
        utt.complete = True
        self.finished += 1
        self._keep.append(utt)
        self._done.append(utt)
        while len(self._done) > self.max_queued:
            self._done.popleft()
            self.dropped_queued += 1


# ======================================================================
# 扇出与落盘
# ======================================================================
class BroadcastDispatcher(ISegmentDispatcher):
    """把同一段广播给多个出口（装配器 + 打印 + 队列……）。

    返回 ``all``：任何一个下游拒收都告诉状态机「没吃下」，让背压统计保持诚实。
    """

    def __init__(self, sinks: Sequence[ISegmentDispatcher]) -> None:
        if not sinks:
            raise ValueError("BroadcastDispatcher 至少需要一个下游")
        self.sinks = list(sinks)

    def dispatch(self, seg: Segment) -> bool:
        return all(s.dispatch(seg) for s in self.sinks)


class UtteranceWriter:
    """整句落盘：wav（整句 + 裁剪版）+ ``manifest.jsonl``（可选内嵌逐帧细节数组）。

    在**消费线程**里调用（:meth:`UtteranceAssembler.take` 之后），不要挂在
    ``on_complete`` 上——那是采集线程，写盘会把实时链路拖住。
    """

    def __init__(self, out_dir: str, frames: bool = False, frame_ms: int = 32,
                 pad_ms: int = 40, keep_trimmed: bool = True) -> None:
        self.out_dir = out_dir
        self.frames = frames
        self.frame_ms = frame_ms
        self.pad_ms = pad_ms
        self.keep_trimmed = keep_trimmed
        self.written = 0
        self._lock = threading.Lock()

    def write(self, utt: Utterance) -> Dict[str, object]:
        sub = os.path.join(self.out_dir, utt.tag) if utt.tag else self.out_dir
        os.makedirs(sub, exist_ok=True)
        stem = "utt%06d_%d-%d_%s" % (utt.utterance_id, utt.start_ms, utt.end_ms,
                                     utt.close_reason or "open")
        paths = {"full": os.path.join(sub, stem + ".wav")}
        _write_wav(paths["full"], utt.pcm, utt.sample_rate)
        trim: Optional[array.array] = None
        if self.keep_trimmed:
            trim = utt.trim_to_speech(pad_ms=self.pad_ms)
            paths["trim"] = os.path.join(sub, stem + "_trim.wav")
            _write_wav(paths["trim"], trim.tobytes(), utt.sample_rate)
        rec = utt.manifest(with_frames=self.frames, frame_ms=self.frame_ms)
        rec["files"] = paths
        rec["trim_samples"] = len(trim) if trim is not None else None
        manifest = os.path.join(sub, "manifest.jsonl")
        with self._lock:
            with open(manifest, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.written += 1
        return rec

    def __call__(self, utt: Utterance) -> Dict[str, object]:
        return self.write(utt)


def _write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
