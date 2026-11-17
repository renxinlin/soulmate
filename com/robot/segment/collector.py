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

四个数据出口
------------
1. :meth:`Utterance.pcm` / :meth:`Utterance.to_int16` / :meth:`Utterance.to_numpy`
   —— 整句波形数组；
2. :meth:`Utterance.frame_detail` —— 与 ``block_ms`` 同栅格的**逐帧细节数组**
   （每帧 rms/peak 电平、是否落在 VAD 语音区间、是否落在补零空洞）；
3. :meth:`Utterance.trim_to_speech` —— 按 VAD 的 ``speech_start_ms`` /
   ``last_speech_ms`` 裁掉首尾静音，这才是真正该送 ASR 的那一段。
4. :meth:`Utterance.to_bundle` —— 上面三个出口**一次打包**成
   :class:`UtteranceBundle`：整句 / 语音两个 :class:`AudioView` + 判决与完整性元信息
   + ``usable_for`` 任务可用性判定，直接交给语音分离 / ASR / 声纹（文档 §5.6.1）。

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
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .dispatcher import ISegmentDispatcher
from .models import CloseReason, SampleStatus, Segment

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
# 交付给下游任务的载荷：一次解码、多视图
# ======================================================================
@dataclass(frozen=True)
class AudioView:
    """一段可以直接喂给模型的音频：int16 单声道样本 + 它在会话时间轴上的位置。

    ``samples`` 是整句波形的**切片副本**，视图交出后采集线程不会再动它，
    因此可以安全跨线程传递、持有、送模型；但它是可变的 ``array('h')``，
    约定**只读**——就地修改不会回写到整句，需要预处理（降噪/重采样）请另建数组。

    时间换算一律走 :meth:`ms_to_index` / :meth:`index_to_ms`，
    两者互逆且都按样本数取整，所以「数组下标 ↔ 绝对毫秒」在整条链路上唯一。
    """

    source: str                                  # "full" 整句 | "speech" 裁到语音
    samples: array.array                         # int16 mono
    sample_rate: int
    start_ms: int                                # 视图起点的会话绝对毫秒
    zeroed_spans_ms: Tuple[Tuple[int, int], ...] = ()   # 落在本视图内的补零区间（绝对毫秒）

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms

    @property
    def duration_ms(self) -> int:
        return len(self.samples) * 1000 // self.sample_rate

    @property
    def pcm(self) -> bytes:
        """int16 LE 字节流（写 wav / 喂只认 bytes 的接口时用）。"""
        return self.samples.tobytes()

    def to_numpy(self, normalize: bool = True):
        """模型输入最常见的形态：``float32``、满量程归一到 ±1.0。

        numpy 惰性导入：不用它的调用方（纯 stdlib 链路、单测）不受影响。
        """
        import numpy as np

        a = np.frombuffer(self.pcm, dtype=np.int16)
        return a.astype(np.float32) / 32768.0 if normalize else a

    def level_db(self) -> Tuple[float, float]:
        """整段 ``(rms_db, peak_db)``，用于快速判断这段值不值得跑模型。"""
        return _levels(self.samples)

    def ms_to_index(self, abs_ms: int) -> int:
        idx = (abs_ms - self.start_ms) * self.sample_rate // 1000
        return max(0, min(self.n_samples, idx))

    def index_to_ms(self, idx: int) -> int:
        return self.start_ms + idx * 1000 // self.sample_rate

    def slice_ms(self, lo_ms: int, hi_ms: int) -> "AudioView":
        """按绝对毫秒再切一刀（分离模型常要求固定长度窗口时用）。"""
        i0, i1 = self.ms_to_index(lo_ms), self.ms_to_index(hi_ms)
        if i1 <= i0:
            raise ValueError("切片为空：%d~%dms 不在 [%d,%d) 内"
                             % (lo_ms, hi_ms, self.start_ms, self.end_ms))
        return AudioView(source=self.source + "[%d,%d]" % (lo_ms, hi_ms),
                         samples=self.samples[i0:i1], sample_rate=self.sample_rate,
                         start_ms=self.index_to_ms(i0),
                         zeroed_spans_ms=_clip_spans(self.zeroed_spans_ms, lo_ms, hi_ms))

    def has_zeroed(self, lo_ms: Optional[int] = None, hi_ms: Optional[int] = None) -> bool:
        """区间内是否含补零（即非真实采集的音频）；不给参数就是整段。"""
        if lo_ms is None and hi_ms is None:
            return bool(self.zeroed_spans_ms)
        lo = self.start_ms if lo_ms is None else lo_ms
        hi = self.end_ms if hi_ms is None else hi_ms
        return any(x < hi and y > lo for x, y in self.zeroed_spans_ms)

    def as_dict(self) -> Dict[str, object]:
        """元信息（**不含波形**），可直接进 JSONL。"""
        return {"source": self.source, "n_samples": self.n_samples,
                "start_ms": self.start_ms, "end_ms": self.end_ms,
                "duration_ms": self.duration_ms, "sample_rate": self.sample_rate,
                "zeroed_spans_ms": [list(x) for x in self.zeroed_spans_ms]}


@dataclass(frozen=True)
class UtteranceBundle:
    """一句完整话的**交付载荷**：波形 + 语音段 + 判决与完整性元信息 + 任务可用性判定。

    设计取向（给语音分离 / ASR / 声纹这类下游任务用）：

    * **一次解码**：:meth:`Utterance.to_bundle` 只把整句 PCM 解一次，两个视图都从
      同一份数组切出来；调用方再取 `pcm` / `to_numpy` 都是在这份数据上加工，
      不会重复付 O(N) 的代价。
    * **两个视图**：`full` 是含 pre/post_roll 与首尾静音的整句（分离模型通常要这个，
      上下文静音对它有用）；`speech` 是按 VAD 边界裁过的（ASR / 声纹要的）。
    * **诚实的完整性**：`suspect` / `suspect_reasons` / `usable_for` 把「这段音频能不能
      信」讲明白。空洞一律补零以保住时间映射，但补零处不是真声，声纹这种对伪影
      敏感的任务必须先看这个字段再跑。
    * 判定规则写在 :meth:`Utterance.to_bundle` 里，是**下限式**的：只在明显不该跑时拦，
      拿不准就放行并把原因挂在 `usable_for` 上，别让下游任务对着一个布尔值猜。
    """

    utterance_id: int
    tag: str
    sample_rate: int
    channels: int
    full: AudioView
    speech: AudioView
    start_ms: int
    end_ms: int
    duration_ms: int
    speech_start_ms: int
    speech_end_ms: int
    speech_duration_ms: int
    n_parts: int
    seq_first: int
    seq_last: int
    complete: bool
    aborted: bool
    close_reason: Optional[str]
    contiguous: bool
    hole_ms: int
    holes: Tuple[Hole, ...]
    zeroed_spans_ms: Tuple[Tuple[int, int], ...]
    suspect: bool
    suspect_reasons: Tuple[str, ...]
    vad_confidence: float
    peak_level_db: float
    mean_level_db: float
    noise_floor_db: float
    decision_ms: int
    usable_for: Dict[str, str] = field(default_factory=dict)
    frames: Optional[Tuple[FrameDetail, ...]] = None

    def audio(self, which: str = "full") -> AudioView:
        """``which="full"`` 整句 / ``"speech"`` 语音段；别的值直接报错，不静默兜底。"""
        if which == "full":
            return self.full
        if which == "speech":
            return self.speech
        raise ValueError("音频视图只有 full / speech，收到 %r" % (which,))

    def usable(self, task: str) -> bool:
        """该任务是否可以拿这份载荷直接开跑。"""
        return self.usable_for.get(task) == "ok"

    def as_dict(self, with_frames: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "utterance_id": self.utterance_id, "tag": self.tag or None,
            "sample_rate": self.sample_rate, "channels": self.channels,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "speech_span_ms": [self.speech_start_ms, self.speech_end_ms],
            "speech_duration_ms": self.speech_duration_ms,
            "n_parts": self.n_parts, "seq": [self.seq_first, self.seq_last],
            "complete": self.complete, "aborted": self.aborted,
            "close_reason": self.close_reason,
            "contiguous": self.contiguous, "hole_ms": self.hole_ms,
            "holes": [{"from_seg": h.from_seg, "to_seg": h.to_seg, "gap_ms": h.gap_ms}
                      for h in self.holes],
            "zeroed_spans_ms": [list(x) for x in self.zeroed_spans_ms],
            "suspect": self.suspect, "suspect_reasons": list(self.suspect_reasons),
            "vad_confidence": self.vad_confidence,
            "peak_level_db": round(self.peak_level_db, 1),
            "mean_level_db": round(self.mean_level_db, 1),
            "noise_floor_db": self.noise_floor_db,
            "decision_ms": self.decision_ms,
            "usable_for": dict(self.usable_for),
            "views": {"full": self.full.as_dict(), "speech": self.speech.as_dict()},
        }
        if with_frames and self.frames is not None:
            out["frames"] = [f.as_dict() for f in self.frames]
        return out

    def json(self, with_frames: bool = False) -> str:
        return json.dumps(self.as_dict(with_frames), ensure_ascii=False)

    def summary(self) -> str:
        return ("utt#%d%s [%d,%d] %dms 语音%dms 片%d %s%s 可用性%s" % (
            self.utterance_id, ("@" + self.tag) if self.tag else "",
            self.start_ms, self.end_ms, self.duration_ms, self.speech_duration_ms,
            self.n_parts, self.close_reason or "open",
            "/aborted" if self.aborted else "",
            " ".join("%s:%s" % (k, v) for k, v in sorted(self.usable_for.items())) or "-"))

    def __str__(self) -> str:
        return self.summary()


def _clip_spans(spans: Tuple[Tuple[int, int], ...], lo_ms: int,
                hi_ms: int) -> Tuple[Tuple[int, int], ...]:
    """把补零区间裁到 ``[lo_ms, hi_ms)`` 内；完全落在外面的丢掉。"""
    out = []
    for x, y in spans:
        a, b = max(x, lo_ms), min(y, hi_ms)
        if b > a:
            out.append((a, b))
    return tuple(out)


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

    # ------------------------------------------------------------------ 交付载荷
    def to_bundle(self, pad_ms: int = 40, frame_ms: int = 32, with_frames: bool = False,
                  min_voiceprint_ms: int = 1000) -> UtteranceBundle:
        """打包整句：完整波形 + 语音段 + 元信息 + 各下游任务的可用性判定。

        用途就是把「一句完整话」一次性交给语音分离 / ASR / 声纹，调用方不用再
        自己拼 ``trim_to_speech`` + ``manifest`` + 空洞判断这套重复代码。

        Parameters
        ----------
        pad_ms            : 语音段视图在 VAD 边界外保留的余量（给辅音头尾，别用 0）。
        frame_ms          : 逐帧细节栅格；传 ``SegmentConfig.block_ms`` 才能对回状态机帧序。
        with_frames       : 是否顺带算好逐帧细节数组（声纹/分离想做帧级筛选用）。
        min_voiceprint_ms : 语音短于它就不跑声纹——短音频的说话人嵌入极不稳。

        视图切法
        --------
        ``full``   = ``[parts[0].start_ms, parts[-1].end_ms)``，含 pre/post_roll、含补零空洞，
                     分离模型要的「带上下文的整句」。
        ``speech`` = ``[speech_start - pad, speech_end + pad]`` 再夹到整句范围内，
                     VAD 判出的语音外包络，ASR / 声纹要的「只有人声的那段」。
                     **没有任何 VAD 语音时它是空的**（不是退回整句），
                     这样下游看 ``speech_duration_ms == 0`` 就知道该跳过。

        可用性判定（下限式：只在明显不该跑时拦，其余把原因挂在字段上）
        --------------------------------------------------------------
        separation : ``ok`` 需无正空洞（补零会让分离模型凭空造一个「源」）且整句非空；
                     否则 ``hole_+Nms`` / ``empty``。被截断不影响——分离本来就按窗口算。
        asr        : ``ok`` 需语音视图非空；否则 ``no_speech``。
                     截断/空洞**不拦**：有多少字转多少字，比整句丢掉好，
                     真伪靠 ``suspect`` / ``zeroed_spans_ms`` 自查。
        voiceprint : ``ok`` 需 ``speech_duration_ms >= min_voiceprint_ms``
                     且 ``not suspect``（干净、非截断、语音区间内无补零）；
                     否则 ``short_Nms`` / 首个 suspect 原因。

        只对**已收尾**的整句调用（:meth:`UtteranceAssembler.take` 拿到的那些）：
        对在途句子（:meth:`UtteranceAssembler.current`）打包会和采集线程的
        ``append`` 抢同一份波形，读到的可能是半截 PCM。
        """
        sr = self.sample_rate
        samples = self.to_int16()                                  # 全句只解码这一次
        spans = tuple(self._zero_spans)                            # 绝对毫秒
        u0, u1 = self.start_ms, self.end_ms
        full = AudioView("full", samples, sr, u0, _clip_spans(spans, u0, u1))

        voice = self.speech_spans_ms()
        if voice:
            i0 = self.ms_to_index(voice[0][0] - pad_ms)
            i1 = self.ms_to_index(voice[-1][1] + pad_ms)
            sp_lo, sp_hi = self.speech_span_ms()
        else:
            i0 = i1 = 0
            sp_lo = sp_hi = 0
        if i1 < i0:                                                # pad 把区间夹反时兜底
            i0 = i1 = 0
        s_start = self.index_to_ms(i0)
        s_end = self.index_to_ms(i1)
        speech = AudioView("speech", samples[i0:i1], sr, s_start,
                           _clip_spans(spans, s_start, s_end) if i1 > i0 else ())

        pos_holes = tuple(h.gap_ms for h in self.holes if h.gap_ms > 0)
        overlaps = tuple(-h.gap_ms for h in self.holes if h.gap_ms < 0)
        reasons: List[str] = []
        for s in self.parts:
            if s.status is not SampleStatus.CLEAN:
                reasons.append("seg#%d_%s" % (s.seg_id, s.status.value))
        if pos_holes:
            reasons.append("hole_+%dms" % sum(pos_holes))
        if overlaps:
            reasons.append("overlap_%dms" % sum(overlaps))
        if self.aborted:
            reasons.append("aborted")
        if not self.complete:
            reasons.append("incomplete")
        if speech.has_zeroed():
            reasons.append("zeroed_in_speech_%dms"
                           % sum(y - x for x, y in speech.zeroed_spans_ms))
        suspect = bool(reasons)

        parts = self.parts
        dur_all = sum(s.duration_ms for s in parts) or 1
        peak = max([float(s.meta.get("peak_level_db", -120.0)) for s in parts] or [-120.0])
        mean = sum(float(s.meta.get("mean_level_db", -120.0)) * s.duration_ms
                   for s in parts) / dur_all
        conf = sum(s.vad_confidence * s.duration_ms for s in parts) / dur_all
        if speech.duration_ms < min_voiceprint_ms:
            voiceprint = "short_%dms" % speech.duration_ms
        elif suspect:
            voiceprint = reasons[0]
        else:
            voiceprint = "ok"
        usable = {
            "separation": "ok" if (full.n_samples and not pos_holes) else (
                "empty" if not full.n_samples else "hole_+%dms" % sum(pos_holes)),
            "asr": "ok" if speech.n_samples else "no_speech",
            "voiceprint": voiceprint,
        }

        return UtteranceBundle(
            utterance_id=self.utterance_id, tag=self.tag, sample_rate=sr, channels=1,
            full=full, speech=speech, start_ms=u0, end_ms=u1,
            duration_ms=self.duration_ms,
            speech_start_ms=sp_lo, speech_end_ms=sp_hi,
            speech_duration_ms=speech.duration_ms,
            n_parts=len(parts),
            seq_first=parts[0].seq_in_utterance if parts else -1,
            seq_last=parts[-1].seq_in_utterance if parts else -1,
            complete=self.complete, aborted=self.aborted, close_reason=self.close_reason,
            contiguous=self.contiguous, hole_ms=self.hole_ms, holes=tuple(self.holes),
            zeroed_spans_ms=spans, suspect=suspect, suspect_reasons=tuple(reasons),
            vad_confidence=round(conf, 3), peak_level_db=peak, mean_level_db=round(mean, 1),
            noise_floor_db=parts[-1].noise_floor_db if parts else -120.0,
            decision_ms=parts[-1].decision_ms if parts else 0,
            usable_for=usable,
            frames=tuple(self.frame_detail(frame_ms)) if with_frames else None,
        )

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
