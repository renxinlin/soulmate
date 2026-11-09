# 多说话人分离与识别流水线 — 设计文档

## 1. 系统概述

本流水线完成 **多说话人混合音频** 的全链路处理：
- 有 DOA（多通道）：预处理 → 波束形成 → 已知说话人提取 → 未知说话人分离 → ASR
- 无 DOA（单通道）：预处理 → 声纹匹配 → 目标说话人提取 → 未知聚类 → ASR

最终输出每个说话人的 ASR 识别结果（含置信度过滤）。

---

## 2. 流程拓扑

```
[MixedAudio]
    │
    ▼
┌──────────────────┐
│AudioPreprocessor │  分帧 + VAD
└────────┬─────────┘
         │
         ├── mono ──► (跳过 DOA / Beamformer)
         │
         ▼
┌──────────────────┐
│ DOAEstimator     │  估计 M 个方向
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Beamformer       │  生成 M 个波束
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│VoiceprintExtractor│  每波束提取声纹特征
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│SpeakerMatcher    │  与已知说话人库匹配
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
 存在已知   无已知
    │         │
    ▼         │
┌──────────────────┐  │
│TargetSpeaker     │  │  对每个已知说话人做分离
│Extractor         │  │
└────────┬─────────┘  │
         │            │
         ▼            │
┌──────────────────┐  │
│ResidualCalculator│  │  混合 − Σ已知分离
└────────┬─────────┘  │
         │            │
         ▼            ▼
┌──────────────────┐
│UnknownSpeaker    │  未知声纹匹配 + 聚类
│Clusterer         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ASREngine        │  每路聚类音频 → ASR
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ConfidenceFilter  │  丢弃低置信度路
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ResultAggregator  │  汇总输出
└──────────────────┘
```

---

## 3. 数据模型（DTO）

### 3.1 核心数据结构

```python
# com/robot/speaker/pipeline/models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ChannelType(Enum):
    """音频通道类型。"""
    MONO = "mono"       # 单通道 → 跳过 DOA/Beamform
    MULTI = "multi"     # 多通道阵列 → 全流水线


@dataclass
class AudioFrame:
    """单帧音频。"""
    samples: np.ndarray          # shape (num_channels, frame_size)
    frame_index: int
    timestamp: float             # 帧起始时间（秒）
    is_speech: bool = False      # VAD 标注


@dataclass
class AudioSegment:
    """一段连续语音。"""
    samples: np.ndarray          # shape (num_channels, num_samples)
    start_time: float
    end_time: float
    frames: List[AudioFrame] = field(default_factory=list)

    @property
    def num_channels(self) -> int:
        return self.samples.shape[0] if self.samples.ndim > 1 else 1


@dataclass
class MixedAudio:
    """流水线输入：混合音频。"""
    samples: np.ndarray          # (num_channels, num_samples)
    sample_rate: int
    channel_type: ChannelType
    num_channels: int = 1

    def __post_init__(self):
        if self.samples.ndim == 1:
            self.channel_type = ChannelType.MONO
            self.num_channels = 1
        else:
            self.num_channels = self.samples.shape[0]
            self.channel_type = ChannelType.MULTI if self.num_channels > 1 else ChannelType.MONO


@dataclass
class DOAResult:
    """DOA 估计结果。"""
    angles: List[float]          # M 个方位角（度），[-90, 90] or [0, 360]
    powers: List[float]          # 各方向能量
    confidence: List[float]      # 各方向置信度


@dataclass
class Beam:
    """单个波束输出。"""
    doa_index: int
    angle: float
    samples: np.ndarray          # (num_samples,) 波束形成后单路音频


@dataclass
class VoiceprintEmbedding:
    """声纹特征向量。"""
    vector: np.ndarray           # (feature_dim,)
    beam_index: Optional[int] = None


@dataclass
class SpeakerMatch:
    """说话人匹配结果。"""
    speaker_id: str              # 已知说话人 ID
    similarity: float            # 相似度 [0, 1]
    beam_index: Optional[int] = None


@dataclass
class ExtractedSpeaker:
    """目标说话人提取输出。"""
    speaker_id: str
    audio: AudioSegment
    similarity: float


@dataclass
class ClusteredSpeaker:
    """聚类后的未知说话人。"""
    cluster_id: str              # 自动生成 ID，如 "unknown_0"
    audio: AudioSegment
    embedding: Optional[VoiceprintEmbedding] = None
    size: int = 1                # 该聚类包含的帧数


@dataclass
class ASRResult:
    """单路 ASR 识别结果。"""
    text: str
    confidence: float            # [0, 1]
    speaker_id: str
    speaker_type: str            # "known" / "unknown"


@dataclass
class PipelineOutput:
    """流水线最终输出。"""
    results: List[ASRResult]
    total_speakers: int
    processing_time: float       # 秒
    metadata: Dict = field(default_factory=dict)
```

---

## 4. 接口定义（抽象基类）

### 4.1 AudioPreprocessor

```python
# com/robot/speaker/pipeline/interfaces/preprocessor.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioFrame, AudioSegment, MixedAudio


class IAudioPreprocessor(ABC):
    """
    音频预处理器：分帧 + VAD 标注。

    职责：
      - 将混合音频按帧切分（frame_size / hop_size 可配置）
      - 对每帧执行语音活动检测，标注 is_speech
      - 提取连续语音段 (AudioSegment)
    """

    @abstractmethod
    def frame(self, audio: MixedAudio,
              frame_size: int = 512,
              hop_size: int = 256) -> List[AudioFrame]:
        """分帧。返回所有帧列表（含 VAD 标注）。"""

    @abstractmethod
    def extract_speech_segments(self,
                                frames: List[AudioFrame]) -> List[AudioSegment]:
        """根据 VAD 标注提取连续语音段。"""

    @abstractmethod
    def preprocess(self, audio: MixedAudio) -> List[AudioSegment]:
        """完整预处理入口：分帧 → VAD → 提取语音段。"""
```

### 4.2 DOAEstimator

```python
# com/robot/speaker/pipeline/interfaces/doa.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, DOAResult


class IDOAEstimator(ABC):
    """
    方向到达（DOA）估计器。

    职责：
      - 从多通道语音段估计 M 个声源方向角
      - 单通道模式下整模块跳过

    典型实现：MUSIC / SRP-PHAT / GCC-PHAT 聚类
    """

    @abstractmethod
    def estimate(self, segments: List[AudioSegment],
                 num_speakers: int = 0) -> DOAResult:
        """
        估计声源方向。

        Parameters
        ----------
        segments     : 预处理后的语音段
        num_speakers : 预设说话人数，0 表示自适应

        Returns
        -------
        DOAResult: 含 M 个方向角、能量和置信度
        """
```

### 4.3 Beamformer

```python
# com/robot/speaker/pipeline/interfaces/beamformer.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, Beam, DOAResult, MixedAudio


class IBeamformer(ABC):
    """
    多波束形成器。

    职责：
      - 根据 M 个 DOA 生成 M 个空间波束
      - 每个波束聚焦一个方向，抑制其他方向

    典型实现：Delay-and-Sum / MVDR / GSC
    """

    @abstractmethod
    def form_beams(self, audio: MixedAudio,
                   doa_result: DOAResult) -> List[Beam]:
        """生成所有波束。"""

    @abstractmethod
    def beam_to_segment(self, beam: Beam,
                        sample_rate: int) -> AudioSegment:
        """将波束输出封装为 AudioSegment，供下游使用。"""
```

### 4.4 VoiceprintExtractor

```python
# com/robot/speaker/pipeline/interfaces/voiceprint.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, Beam, VoiceprintEmbedding


class IVoiceprintExtractor(ABC):
    """
    声纹特征提取器。

    职责：
      - 从单路音频提取固定维度的声纹嵌入向量
      - 支持对每个波束分别提取

    典型实现：ECAPA-TDNN / d-vector / x-vector
    """

    @abstractmethod
    def extract(self, segment: AudioSegment) -> VoiceprintEmbedding:
        """提取声纹特征。"""

    @abstractmethod
    def extract_batch(self, segments: List[AudioSegment]) -> List[VoiceprintEmbedding]:
        """批量提取（对 M 个波束）。"""

    @abstractmethod
    def feature_dim(self) -> int:
        """返回特征向量维度。"""
```

### 4.5 SpeakerMatcher

```python
# com/robot/speaker/pipeline/interfaces/matcher.py

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import SpeakerMatch, VoiceprintEmbedding


class ISpeakerMatcher(ABC):
    """
    说话人匹配器：将声纹向量与已知说话人库比对。

    职责：
      - 加载/管理已知说话人声纹库
      - 对给定声纹向量返回最佳匹配（超过阈值视为已知）

    典型实现：cosine similarity + threshold / PLDA
    """

    @abstractmethod
    def register(self, speaker_id: str,
                 embedding: VoiceprintEmbedding) -> None:
        """注册已知说话人。"""

    @abstractmethod
    def match(self, embedding: VoiceprintEmbedding,
              top_k: int = 1) -> List[SpeakerMatch]:
        """声纹匹配，返回 top_k 结果。"""

    @abstractmethod
    def is_known(self, embedding: VoiceprintEmbedding,
                 threshold: float = 0.75) -> Optional[SpeakerMatch]:
        """判断是否存在已知说话人（单结果）。"""

    @abstractmethod
    def match_batch(self, embeddings: List[VoiceprintEmbedding]
                    ) -> List[Optional[SpeakerMatch]]:
        """批量匹配（M 个波束 → M 个结果）。"""
```

### 4.6 TargetSpeakerExtractor

```python
# com/robot/speaker/pipeline/interfaces/tse.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, ExtractedSpeaker, SpeakerMatch


class ITargetSpeakerExtractor(ABC):
    """
    目标说话人提取器（TSE）。

    职责：
      - 给定混合音频 + 目标说话人声纹/ID
      - 从混合中分离出该说话人的纯净音频

    典型实现：Conv-TasNet / SpEx / VoiceFilter / TSE based on speaker embedding
    """

    @abstractmethod
    def extract(self, mixed: AudioSegment,
                target: SpeakerMatch) -> ExtractedSpeaker:
        """提取单个目标说话人。"""

    @abstractmethod
    def extract_multi(self, mixed: AudioSegment,
                      targets: List[SpeakerMatch]) -> List[ExtractedSpeaker]:
        """提取多个已知说话人。"""
```

### 4.7 ResidualCalculator

```python
# com/robot/speaker/pipeline/interfaces/residual.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, ExtractedSpeaker


class IResidualCalculator(ABC):
    """
    残余音频计算器。

    职责：
      - 从混合音频中减去所有已知说话人的分离音频
      - 输出包含未知说话人的残余
    """

    @abstractmethod
    def compute(self, mixed: AudioSegment,
                extracted: List[ExtractedSpeaker]) -> AudioSegment:
        """
        residual = mixed - Σ extracted[i].audio

        对齐策略：按样本对齐，短音频补零。
        """
```

### 4.8 UnknownSpeakerClusterer

```python
# com/robot/speaker/pipeline/interfaces/clusterer.py

from abc import ABC, abstractmethod
from typing import List

from ..models import AudioSegment, ClusteredSpeaker, VoiceprintEmbedding


class IUnknownSpeakerClusterer(ABC):
    """
    未知说话人聚类器。

    职责：
      - 对残余音频做声纹匹配
      - 按说话人聚类，输出 N 路独立说话人音频

    典型实现：Agglomerative Clustering / Spectral Clustering on embeddings
    """

    @abstractmethod
    def cluster(self, residual: AudioSegment,
                max_clusters: int = 5,
                threshold: float = 0.6) -> List[ClusteredSpeaker]:
        """对残余音频聚类。"""

    @abstractmethod
    def match_and_cluster(self, residual: AudioSegment,
                          embeddings: List[VoiceprintEmbedding]
                          ) -> List[ClusteredSpeaker]:
        """已知声纹匹配 + 剩余聚类。"""
```

### 4.9 ASREngine

```python
# com/robot/speaker/pipeline/interfaces/asr.py

from abc import ABC, abstractmethod
from typing import List

from ..models import ASRResult, AudioSegment


class IASREngine(ABC):
    """
    语音识别引擎。

    职责：
      - 对每路音频做 ASR
      - 输出识别文本 + 置信度

    典型实现：FunASR / Whisper / Paraformer
    """

    @abstractmethod
    def recognize(self, segment: AudioSegment,
                  speaker_id: str = "",
                  speaker_type: str = "unknown") -> ASRResult:
        """单路识别。"""

    @abstractmethod
    def recognize_batch(self, segments: List[AudioSegment],
                        speaker_ids: List[str],
                        speaker_types: List[str]) -> List[ASRResult]:
        """批量识别。"""
```

### 4.10 ConfidenceFilter

```python
# com/robot/speaker/pipeline/interfaces/filter.py

from abc import ABC, abstractmethod
from typing import List

from ..models import ASRResult


class IConfidenceFilter(ABC):
    """
    ASR 置信度过滤器。

    职责：
      - 丢弃置信度低于阈值的路
      - 保留高置信度结果
    """

    @abstractmethod
    def filter(self, results: List[ASRResult],
               threshold: float = 0.6) -> List[ASRResult]:
        """过滤低置信度结果。"""

    @abstractmethod
    def filter_adaptive(self, results: List[ASRResult]) -> List[ASRResult]:
        """自适应阈值过滤。"""
```

### 4.11 ResultAggregator

```python
# com/robot/speaker/pipeline/interfaces/aggregator.py

from abc import ABC, abstractmethod
from typing import List

from ..models import ASRResult, PipelineOutput


class IResultAggregator(ABC):
    """
    结果汇总器。

    职责：
      - 收集过滤后的 ASR 结果
      - 按说话人/时间排序
      - 组装 PipelineOutput
    """

    @abstractmethod
    def aggregate(self, results: List[ASRResult],
                  processing_time: float) -> PipelineOutput:
        """汇总最终输出。"""
```

---

## 5. 流水线编排器

### 5.1 SpeakerPipeline

```python
# com/robot/speaker/pipeline/orchestrator.py

from typing import List, Optional

from .interfaces import (
    IAudioPreprocessor, IBeamformer, IConfidenceFilter,
    IDAEstimator, IResidualCalculator, IResultAggregator,
    ISpeakerMatcher, ITargetSpeakerExtractor,
    IUnknownSpeakerClusterer, IVoiceprintExtractor, IASREngine,
)
from .models import (
    ASRResult, AudioSegment, Beam, ClusteredSpeaker, DOAResult,
    ExtractedSpeaker, MixedAudio, PipelineOutput,
    SpeakerMatch, VoiceprintEmbedding,
)


class SpeakerPipeline:
    """
    多说话人流水线编排器。

    将各模块按拓扑顺序串联，根据 channel_type 自动跳过 DOA/Beamform 步骤。

    Parameters
    ----------
    preprocessor     : IAudioPreprocessor    分帧 + VAD
    doa_estimator    : IdoaEstimator | None  DOA 估计（单通道为 None）
    beamformer       : IBeamformer | None    波束形成（单通道为 None）
    voiceprint_ext   : IVoiceprintExtractor  声纹提取
    speaker_matcher  : ISpeakerMatcher       说话人匹配
    tse_extractor    : ITargetSpeakerExtractor  目标说话人提取
    residual_calc    : IResidualCalculator   残余计算
    clusterer        : IUnknownSpeakerClusterer  未知聚类
    asr_engine       : IASREngine            语音识别
    conf_filter      : IConfidenceFilter     置信度过滤
    aggregator       : IResultAggregator     结果汇总
    """

    def __init__(
        self,
        preprocessor: IAudioPreprocessor,
        doa_estimator: Optional[IDOAEstimator],
        beamformer: Optional[IBeamformer],
        voiceprint_ext: IVoiceprintExtractor,
        speaker_matcher: ISpeakerMatcher,
        tse_extractor: ITargetSpeakerExtractor,
        residual_calc: IResidualCalculator,
        clusterer: IUnknownSpeakerClusterer,
        asr_engine: IASREngine,
        conf_filter: IConfidenceFilter,
        aggregator: IResultAggregator,
        min_confidence: float = 0.6,
    ) -> None:
        ...

    def process(self, audio: MixedAudio) -> PipelineOutput:
        """流水线主入口：处理混合音频，返回最终识别结果。"""
        ...

    # 内部步骤方法（可被子类覆写）
    def _step_preprocess(self, audio: MixedAudio) -> List[AudioSegment]: ...
    def _step_doa(self, segments: List[AudioSegment]) -> Optional[DOAResult]: ...
    def _step_beamform(self, audio: MixedAudio,
                       doa: DOAResult) -> List[Beam]: ...
    def _step_voiceprint(self, beams: List[Beam]) -> List[VoiceprintEmbedding]: ...
    def _step_match(self, embeddings: List[VoiceprintEmbedding]) -> List[SpeakerMatch]: ...
    def _step_extract(self, mixed: AudioSegment,
                      known: List[SpeakerMatch]) -> List[ExtractedSpeaker]: ...
    def _step_residual(self, mixed: AudioSegment,
                       extracted: List[ExtractedSpeaker]) -> AudioSegment: ...
    def _step_cluster(self, residual: AudioSegment) -> List[ClusteredSpeaker]: ...
    def _step_asr(self, speakers: List[ClusteredSpeaker],
                  known: List[ExtractedSpeaker]) -> List[ASRResult]: ...
    def _step_filter(self, results: List[ASRResult]) -> List[ASRResult]: ...
    def _step_aggregate(self, results: List[ASRResult]) -> PipelineOutput: ...
```

---

## 6. 模块目录结构

```
com/robot/speaker/
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── models.py               # 所有 DTO 数据结构
│   ├── orchestrator.py         # SpeakerPipeline 编排器
│   ├── interfaces/             # 抽象基类定义
│   │   ├── __init__.py         # re-export 所有 I* 类
│   │   ├── preprocessor.py     # IAudioPreprocessor
│   │   ├── doa.py              # IdoaEstimator
│   │   ├── beamformer.py       # IBeamformer
│   │   ├── voiceprint.py       # IVoiceprintExtractor
│   │   ├── matcher.py          # ISpeakerMatcher
│   │   ├── tse.py              # ITargetSpeakerExtractor
│   │   ├── residual.py         # IResidualCalculator
│   │   ├── clusterer.py        # IUnknownSpeakerClusterer
│   │   ├── asr.py              # IASREngine
│   │   ├── filter.py           # IConfidenceFilter
│   │   └── aggregator.py       # IResultAggregator
│   └── impl/                   # 具体实现（后续按需填充）
│       ├── __init__.py
│       ├── preprocessor_impl.py    # FSMN-VAD 实现
│       ├── doa_impl.py             # MUSIC / SRP-PHAT
│       ├── beamformer_impl.py      # Delay-and-Sum / MVDR
│       ├── voiceprint_impl.py      # ECAPA-TDNN
│       ├── matcher_impl.py         # cosine + threshold
│       ├── tse_impl.py             # Conv-TasNet / SpEx
│       ├── residual_impl.py        # numpy 减法
│       ├── clusterer_impl.py       # agglomerative clustering
│       ├── asr_impl.py             # FunASR 封装
│       ├── filter_impl.py          # 固定阈值过滤
│       └── aggregator_impl.py      # 简单列表汇总
```

---

## 7. 接口契约汇总

| 步骤 | 接口 | 输入 | 输出 | 跳过条件 |
|------|------|------|------|----------|
| 1 | `IAudioPreprocessor` | `MixedAudio` | `List[AudioSegment]` | — |
| 3 | `IDOAEstimator` | `List[AudioSegment]` | `DOAResult` | `channel_type == MONO` |
| 4 | `IBeamformer` | `MixedAudio + DOAResult` | `List[Beam]` | `channel_type == MONO` |
| 5 | `IVoiceprintExtractor` | `AudioSegment (per beam)` | `VoiceprintEmbedding` | — |
| 6 | `ISpeakerMatcher` | `VoiceprintEmbedding` | `List[SpeakerMatch]` | — |
| 9 | `ITargetSpeakerExtractor` | `AudioSegment + SpeakerMatch` | `ExtractedSpeaker` | `无已知说话人` |
| 11 | `IResidualCalculator` | `Mixed + List[Extracted]` | `AudioSegment` | `无已知说话人` |
| 13-14 | `IUnknownSpeakerClusterer` | `AudioSegment` | `List[ClusteredSpeaker]` | — |
| 15 | `IASREngine` | `AudioSegment` | `ASRResult` | — |
| 17 | `IConfidenceFilter` | `List[ASRResult]` | `List[ASRResult]` | — |
| 18 | `IResultAggregator` | `List[ASRResult]` | `PipelineOutput` | — |

---

## 8. 扩展点

- **并行执行**：步骤 5-9 各波束之间可并行；步骤 15 各路 ASR 可并行。
- **可插拔实现**：所有模块通过接口注入，支持运行时替换。
- **回调钩子**：`SpeakerPipeline` 可注册 `on_step_complete(step_name, result)` 钩子用于监控和调试。
- **单通道降级**：当输入为单通道或无 DOA 估计时，直接用全段音频做声纹匹配和聚类。
