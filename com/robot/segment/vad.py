# -*- coding: utf-8 -*-
"""帧级 VAD 后端与「段输出型 VAD」适配器。

设计文档假定状态机消费**逐帧概率**（``AudioFrame.prob``），但仓库现有原型
``com/robot/models/asr/asr.py`` 用的是 **FunASR FSMN-VAD**，它输出的是
``[[start_ms, end_ms], ...]`` 段列表而不是帧概率。两类后端在本模块统一成
逐帧判决，状态机不需要感知差异（文档 §7-G19）。

后端清单
--------
* :class:`EnergyVad`          —— 纯 stdlib 能量门限 + 自适应底噪，调试 / 兜底 / 单测。
* :class:`ScriptedVad`        —— 从预置概率序列取值，用于 golden 测试与线上回放。
* :class:`SherpaSileroVad`    —— sherpa-onnx silero-vad（readme 的 ASR 选型）。
* :class:`WindowSegmentVad`   —— 把「整窗 → 段列表」模型（FunASR FSMN-VAD）适配成逐帧判决。

``numpy`` / ``funasr`` / ``sherpa_onnx`` 都是**可选依赖**，只在对应后端真正被使用时才导入。
"""
from __future__ import annotations

import array
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# ======================================================================
# 工具
# ======================================================================
def iter_frames(pcm: bytes, sample_rate: int, block_ms: int, start_ms: int = 0) -> Iterable[Tuple[int, int, bytes]]:
    """把 PCM 切成 ``(start_ms, end_ms, block_bytes)``。"""
    block = sample_rate * block_ms // 1000
    step = block * 2
    pos = 0
    t = start_ms
    while pos + step <= len(pcm):
        yield t, t + block_ms, pcm[pos:pos + step]
        pos += step
        t += block_ms


def rms_db(samples: bytes) -> float:
    """int16 PCM 的 RMS 电平（dBFS）；静音返回 -120。"""
    if len(samples) < 2:
        return -120.0
    a = array.array("h")
    a.frombytes(samples[: len(samples) - (len(samples) % 2)])
    if not a:
        return -120.0
    acc = 0
    for v in a:
        acc += v * v
    rms = math.sqrt(acc / len(a))
    if rms <= 1.0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ======================================================================
# 接口
# ======================================================================
class ISegmentVAD(ABC):
    """逐帧 VAD 判决源。"""

    @abstractmethod
    def prob(self, samples: bytes) -> float:
        """喂入一帧 int16 PCM，返回该帧为语音的概率 [0, 1]。"""

    def reset(self) -> None:
        """会话重启 / 时间轴断裂时清空内部状态。"""


# ======================================================================
# 实现：能量
# ======================================================================
class EnergyVad(ISegmentVAD):
    """能量门限 VAD（无第三方依赖）。

    ``p = clamp((level - (thr - span)) / span)``，并带**很短**的释放拖尾：
    拖尾只为防止单帧跌破，静音时长/间隙合并是状态机的职责（文档 §7-G13），
    两处各拖 100ms 以上就会叠加成「说完还要等一会儿」——正是
    ``com/robot/models/asr/asr.py`` 里 FSMN 尾静音与 ``SILENCE_PADDING`` 叠加的问题。
    """

    def __init__(
        self,
        threshold_db: float = -45.0,
        over_noise_db: float = 6.0,
        span_db: float = 8.0,
        release: float = 0.6,
        floor_init_db: float = -60.0,
        floor_ema: float = 0.02,
    ) -> None:
        self.threshold_db = threshold_db
        self.over_noise_db = over_noise_db
        self.span_db = span_db
        self.release = release
        self.floor_init_db = floor_init_db
        self.floor_ema = floor_ema
        self._prev = 0.0
        self._floor = floor_init_db

    def reset(self) -> None:
        self._prev = 0.0
        self._floor = self.floor_init_db

    @property
    def noise_floor_db(self) -> float:
        return self._floor

    def prob(self, samples: bytes) -> float:
        level = rms_db(samples)
        thr = max(self.threshold_db, self._floor + self.over_noise_db)
        raw = _clamp01((level - (thr - self.span_db)) / self.span_db)
        p = max(raw, self._prev * self.release)          # 快起慢落
        self._prev = p
        if p < 0.3:                                      # 仅在安静时段学习底噪
            self._floor += self.floor_ema * (level - self._floor)
        return p


# ======================================================================
# 实现：脚本化（测试 / 回放）
# ======================================================================
class ScriptedVad(ISegmentVAD):
    """按帧序返回预置概率。

    ``probs`` 可为列表（按调用次序）或 ``{帧序号: 概率}`` 映射，缺省为 ``default``。
    """

    def __init__(self, probs: Union[Sequence[float], Dict[int, float]], default: float = 0.0) -> None:
        self._probs = probs
        self._default = default
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def prob(self, samples: bytes) -> float:  # noqa: ARG002 - 接口统一，帧内容不参与
        if isinstance(self._probs, dict):
            v = self._probs.get(self._i, self._default)
        else:
            v = self._probs[self._i] if self._i < len(self._probs) else self._default
        self._i += 1
        return float(v)


# ======================================================================
# 实现：sherpa-onnx silero
# ======================================================================
class SherpaSileroVad(ISegmentVAD):
    """sherpa-onnx silero-vad 后端。

    只取「当前是否处于语音中」这一判决，输出二值 1.0/0.0；段时长约束
    （``min_speech_ms`` / ``min_silence_ms``）由状态机负责，**不要**再依赖模型
    内部的静音参数——两处判决叠在一起会让延迟无法预测（见 :class:`WindowSegmentVad`
    对 FSMN 同类问题的说明）。
    """

    def __init__(self, model_path: str, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        import sherpa_onnx  # 可选依赖：仅在真正使用时导入

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = threshold
        cfg.sample_rate = sample_rate
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        if not hasattr(self._vad, "accept_waveform"):
            raise RuntimeError(
                "当前 sherpa_onnx 版本未提供 accept_waveform 逐帧接口；"
                "请升级 sherpa-onnx，或改用 WindowSegmentVad 适配段列表输出"
            )

    def reset(self) -> None:
        try:
            self._vad.reset()
        except AttributeError:  # pragma: no cover - 老版本无此方法
            pass

    def prob(self, samples: bytes) -> float:
        a = array.array("h")
        a.frombytes(samples[: len(samples) - (len(samples) % 2)])
        self._vad.accept_waveform([v / 32768.0 for v in a])
        return 1.0 if self._vad.is_speech() else 0.0


# ======================================================================
# 实现：整窗段列表 → 逐帧判决（兼容 FunASR FSMN-VAD）
# ======================================================================
class WindowSegmentVad(ISegmentVAD):
    """把「整窗音频 → 语音段列表」的 VAD 适配成逐帧判决。

    对应 ``com/robot/models/asr/asr.py`` 的用法::

        vad_model.generate(input=audio_np, cache={}, fs=SAMPLE_RATE)
        -> [{"value": [[start_ms, end_ms], ...]}]

    行为与原型一致（整窗重算 + 段尾贴窗视为仍在说话），但有三点刻意的差别：

    1. **只吐「新增步长」那几帧的概率**。原型每 ``500ms`` 重算最近 ``2.5s``，
       并把整窗结果都拿去用，导致同一帧语音被反复判决；此处重算次数相同，
       但只把窗**尾部 ``stride_ms``** 对应的帧概率交给状态机，因此
       **每帧恰好拿到一个概率、不重复不遗漏**，代价是概率流有一个
       恒定的 ``lag_ms`` 滞后（见 :attr:`lag_ms`，批量判决绕不开）。
       冗余计算比为 ``window_ms / stride_ms``（默认 1984/512 ≈ 3.9×，优于原型的 5×；
       由此产生的 ``lag_ms = 480ms`` 恰好被 ``vad_lag_ms=500`` 覆盖）。
    2. **段尾贴窗判断（原型的 ``speech_ongoing`` / ``ONGOING_MARGIN``）上移到适配器**，
       状态机只看到统一的帧概率，不需要知道窗、步长这些概念。
    3. **不隐藏模型自带的尾静音**。FSMN-VAD 默认 ``max_end_silence_time ≈ 800ms``
       才吐段尾——这正是原型注释里「说完后 0.8~1.3s 才出文字」的来源。该值以
       :attr:`model_end_silence_ms` 显式记录：状态机的 ``min_silence_ms`` 是在它**之外**
       再加一层，二者之和才是真实闭合延迟。调参时必须知道这件事，否则会误以为
       「300ms 静音就闭合」。
    """

    def __init__(
        self,
        vad_model,
        sample_rate: int = 16000,
        block_ms: int = 32,
        window_ms: int = 1984,         # 62 × 32ms，≈ 需求的 2s（必须落在帧栅格上）
        stride_ms: int = 512,          # 16 × 32ms，≈ 原型的 500ms 步进
        ongoing_margin_ms: int = 250,
        model_end_silence_ms: int = 800,
        to_float: Optional[Callable[[bytes], object]] = None,
    ) -> None:
        if window_ms % block_ms or stride_ms % block_ms:
            raise ValueError("window_ms / stride_ms 必须是 block_ms 的整数倍")
        self.model = vad_model
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.ongoing_margin_ms = ongoing_margin_ms
        self.model_end_silence_ms = model_end_silence_ms
        self.model_calls = 0

        self._block_bytes = sample_rate // 1000 * block_ms * 2
        self._blocks_per_window = window_ms // block_ms
        self._stride_blocks = max(1, stride_ms // block_ms)
        self._window: bytes = b""          # 只保留最近 window_ms 的 PCM
        self._since_run = 0
        self._pending: List[float] = []    # 已算好、待状态机逐帧取用
        self._to_float = to_float

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._window = b""
        self._pending.clear()
        self._since_run = 0

    @property
    def warmed(self) -> bool:
        """整窗是否已填满；未填满前不做判决（原型同样跳过首个窗口）。"""
        return len(self._window) >= self._blocks_per_window * self._block_bytes

    @property
    def lag_ms(self) -> int:
        """概率流相对帧时钟的**固定滞后** = ``stride_ms - block_ms``。

        批量模型只能在攒够一个 stride 时判决，于是第 ``i`` 帧拿到的是
        ``i - stride + block`` 的概率：滞后恒定为 1 个 stride 减 1 帧。
        这正是需求里「VAD 游标在写指针前 0.5 秒」的物理来源——``vad_lag_ms``
        必须 ``>= lag_ms``，否则状态机拿到的概率与它以为的帧时间不一致。
        """
        return self.stride_ms - self.block_ms

    def prob(self, samples: bytes) -> float:
        self._window += samples
        keep = self._blocks_per_window * self._block_bytes
        if len(self._window) > keep:
            self._window = self._window[-keep:]
        self._since_run += len(samples)
        if self._since_run >= self._block_bytes * self._stride_blocks and self.warmed:
            self._since_run = 0
            self._run_window()
        if self._pending:
            return self._pending.pop(0)
        return 0.0

    # ------------------------------------------------------------------
    def _run_window(self) -> None:
        """在整窗上调用模型，只产出窗尾 ``stride_blocks`` 帧的概率。"""
        self.model_calls += 1
        segs = parse_segment_result(
            self.model.generate(input=self._window_input(), cache={}, fs=self.sample_rate)
        )
        n = self._blocks_per_window
        k = self._stride_blocks
        probs = [0.0] * k
        tail_margin_blocks = max(1, self.ongoing_margin_ms // self.block_ms)
        for s_ms, e_ms in segs:
            for j in range(k):
                i = n - k + j                          # 该帧在窗内的块序号
                b_start, b_end = i * self.block_ms, (i + 1) * self.block_ms
                if b_start < e_ms and b_end > s_ms:
                    probs[j] = 1.0
            # 段尾贴窗 → 说话仍在继续：把窗尾若干块置 1，避免状态机误判收尾
            if e_ms >= self.window_ms - self.ongoing_margin_ms:
                for j in range(max(0, k - tail_margin_blocks), k):
                    probs[j] = 1.0
        self._pending.extend(probs)

    def _window_input(self):
        """模型输入：默认转 float32 numpy（FunASR 的常见形式），可用 ``to_float`` 覆盖。"""
        if self._to_float is not None:
            return self._to_float(self._window)
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - 无 numpy 时原样交给模型
            return self._window
        a = np.frombuffer(self._window, dtype=np.int16)
        return a.astype(np.float32) / 32768.0


def parse_segment_result(res) -> List[Tuple[int, int]]:
    """解析 FunASR VAD 输出为 ``[(start_ms, end_ms), ...]``。

    ⚠ FSMN-VAD 的 ``value`` 单位是**毫秒**，不是采样点——``asr.py`` 里
    「★★★ 核心修复」修的就是这个（除错会导致所有段长 < 0.16s，状态机永不进入说话）。
    """
    out: List[Tuple[int, int]] = []
    if not res:
        return out
    for item in res:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not value or not isinstance(value, list):
            continue
        for seg in value:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                s, e = int(seg[0]), int(seg[1])
                if e > s:
                    out.append((s, e))
    return out
