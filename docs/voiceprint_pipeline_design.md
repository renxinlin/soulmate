# 声纹识别模块（voiceprint）— 设计文档

> 状态：**设计稿，未落库**。本文只定义职责、数据结构、接口、参数、判决口径与验收标准，
> `com/robot/voiceprint/` 目录尚不存在。确认后再按 §11 的分期落地。

---

## 0. 需求原始描述（保留原文口径）

1. 新设计一个**声纹识别模块**，读取 `utt.to_bundle()` 的结果（注意：仅设计文档）。
2. 这是一个**新模块**。
3. 能力至少包括：声纹**注册**、**删除**、**更新**、**查找**。
4. 声纹**打标签**能力，以及**基于标签查找**能力。
5. 其余「还需要什么能力」由本文补全（见 §11）。

---

## 1. 设计目标与非目标

### 1.1 目标

| # | 目标 | 判据 |
|---|------|------|
| O1 | 回答「**刚才说话的是谁**」（1:N identify）与「**是不是他**」（1:1 verify） | 已知集内 accuracy、FAR/FRR 可调可测 |
| O2 | 库的**完整生命周期**：注册 / 追加 / 改名改标签 / 停用 / 删除 / 导出导入 | 每个动作都有幂等语义 + 审计记录 |
| O3 | **标签是一等公民**：多标签、命名空间、按标签检索、按标签批量操作 | `find_by_label("关系:家人")` 不需要外部过滤代码 |
| O4 | **判决可解释**：分数、阈值、归一化前后、margin、质量扣减项全部可见 | `result.explain()` 一行文本讲清「凭什么」 |
| O5 | 与 `com/robot/segment` 的交付载荷**单向依赖**：本模块只读 bundle，不反向 import 采集 | bundle 字段契约见 §3 |
| O6 | 核心逻辑**纯标准库**可跑（numpy 只在向量运算处惰性导入），模型后端可插拔，CI / 无模型环境用 `stub` 后端跑通全部单测 | 沿用 segment 的既有约束 |
| O7 | **本地优先**：默认不出网、不存原始音频；撤回即彻底删除 | §10 |
| O8 | RK 级 CPU 上单句识别**不破对话延迟预算** | §12 |

### 1.2 非目标（本文不解决）

* **不做**说话人分离 / 目标说话人提取 / DOA / 波束形成 —— 那是 `docs/speaker_pipeline_design.md`。
* **不做** ASR、不做说话人**属性**识别（性别/年龄/情绪）。属性可作为标签由人工或旁路模型写入，但本模块不自行推断（隐私 + 误判代价，见 §11-C19）。
* **v1 不做**活体检测 / 反欺骗（TTS 拟声、录音回放）。接口预留 `liveness` 字段并写明风险（§11-C14）。
* **不做**云端声纹账号体系与跨设备同步；只保证导出包可迁移（§11-C5）。
* **不做**多人混合语音里的逐人归属——bundle 只保证「一段话」，重叠语音由分离模块负责切干净后再交给本模块。

### 1.3 与 `speaker_pipeline_design.md` 的边界（同时回答 segment 文档 §7-G17）

| | `docs/speaker_pipeline_design.md` | 本文（`com/robot/voiceprint`） |
|---|---|---|
| 问题 | 一段**混合音频**里有谁、各自说了什么 | 一句**已切好的话**是谁说的，以及「谁」这件事的长期管理 |
| 时间尺度 | 单次调用，无状态 | 跨会话、有状态、有库 |
| 输入 | `MixedAudio` / `AudioSegment`（多通道 ndarray） | `UtteranceBundle`（segment 的交付载荷，mono 16k） |
| 关键组件 | DOA、Beamformer、TSE、聚类 | 提取器、**身份库（模板 + 标签）**、判决与阈值治理、审计 |
| 关系 | 两个方向都只由**本模块的适配器**（§5.5）实现：那边的 `ISpeakerMatcher`(§4.5) ← `MatcherAdapter`，那边的 `IVoiceprintExtractor`(§4.4) ← `PipelineVoiceprintAdapter`。`VoiceprintEmbedding` / `SpeakerMatch` 只是本模块向量的薄封装，**向量与判决口径的 source of truth 在本模块** | — |

一句话：**那边是路径，这边是身份与账本。** 单通道分支（那边 §7 表的跳过条件）其实就是「bundle → 本模块」。

---

## 2. 术语与坐标系

| 术语 | 含义 |
|------|------|
| embedding / 声纹向量 | 模型从一段语音抽出的定长向量（192 维量级），**入库前一律 L2 归一化**，此后 cosine = 点积 |
| sample / 样本 | 一条用于描述某说话人的 embedding（来自一句 bundle）。是注册、去重、拆分、审计的**最小单位** |
| template / 模板 | 一个说话人的全部样本 + 质心 + 标签 + 元信息。`speaker_id` 是它的稳定主键 |
| centroid / 质心 | 该模板全部样本（归一化后）的均值再归一化。检索用它 |
| enrollment / 注册 | 让一段语音成为某模板样本的过程，含质量门控与两步确认（§6.1） |
| probe / 探针 | 待识别的那一段（一个 bundle） |
| gallery | 库内全部 active 模板 |
| score | `cos(probe, centroid) ∈ [-1, 1]`；归一化后为 `norm_score` |
| identify / verify | 1:N「是谁」 / 1:1「是不是他」 |
| unknown rejection | 分数不到阈值 ⇒ 判「库里没有这个人」，**不**硬塞给最像的人 |
| cohort / T-norm / Z-norm | 用一批「背景人」的分数把原始 cosine 校准成可比分数，用于阈值跨模板/跨信道对齐（§7.3，v2） |
| EER / minDCF / FAR / FRR | 等错误率 / 最小代价函数 / 误接受率 / 误拒率。阈值由它们定，不靠拍脑袋（§7.2） |
| aging / drift | 跨月后同一人分数系统性下移（设备、年龄、状态），需要监测而非放任 |

**两套时钟，不许混用**：

* bundle 里的 `start_ms / end_ms` 是**会话音频时钟**（segment 文档 §2 的坐标系），只用来定位「这句话在本次采集的第几毫秒」；
* 模板的 `created_at / last_seen_at / updated_at` 用 **UTC 墙钟**（epoch 秒），用来做过期、审计、老化。
  落库字段统一带 `_at`（墙钟）与 `_ms`（会话音频时钟）后缀。

---

## 3. 输入契约：`UtteranceBundle` → 声纹

本节是**唯一的接缝**。上游 `to_bundle()` 已经做了整句拼接与完整性统计，本模块只消费，不再自己拼句。

| bundle 字段 | 本模块用途 | 不满足时 |
|-------------|-----------|----------|
| `speech`（`AudioView`） | **唯一音频输入**。取 `to_numpy()`（float32 ±1.0）喂模型 | `speech.n_samples == 0` ⇒ 直接 `Reject(no_speech)`，不调模型 |
| `full` | 只做**溯源记录**与调试（不参与判决） | — |
| `sample_rate` / `channels` | 必须 `== cfg.sample_rate` / `== 1` | 不等 ⇒ `Verdict.RATE`（`sample_rate_mismatch`）直接拒，**不静默重采样**——SpeechBrain 之类后端会自己重采样选道，那种便利默认值在这里是隐患 |
| `usable_for["voiceprint"]` | 门控总开关：`ok` 才默认准入 | `short_Nms` ⇒ 拒（注册侧开 `allow_short_enroll`、识别侧传 `allow_short=True` 可降级准入并打质量标记）；`suspect` 原因 ⇒ 见下一行 |
| `suspect_reasons` | 逐条白名单化：`overlap_*` 可放（已裁头），`hole_+*` / `zeroed_in_speech_*` / `seg#*_gap` / `seg#*_clamped` 一律拒 | 拒因进 `QualityReport.blocked_by`，可解释 |
| `speech_duration_ms` | 质量分（越接近 `min_voiceprint_ms` 越低） | — |
| `peak_level_db` / `mean_level_db` / `noise_floor_db` | **SNR 门控**：`mean_level_db - noise_floor_db ≥ min_snr_db`；`peak_level_db` 近 0 dBFS ⇒ 削波嫌疑 | 不达 ⇒ 拒或降级 |
| `vad_confidence` | 质量分权重（低置信 ⇒ 边界可能不准，两端各再缩 `vad_trust_ms`） | — |
| `close_reason` / `aborted` / `complete` | `aborted` 或 `close_reason == "max_duration"` ⇒ 句子可能被腰切，**只准用于 identify，不准用于注册**（半句会拉偏质心） | 注册时 `Reject(truncated)` |
| `utterance_id` / `tag` / `seq_first` / `seq_last` / `start_ms` / `end_ms` | 组成**幂等去重键**（§3.2） | 重复 ⇒ `DuplicateSampleError`（除非显式 `on_duplicate="ignore"`） |
| `decision_ms` | 延迟统计（判决完成时刻） | — |
| `frames` | 仅 `with_frames=True` 时做帧级质量分析（v2） | — |

### 3.1 为什么用 `speech` 而不是 `full`

`full` 含 `pre_roll(200ms)` + `post_roll(100ms)` + 首尾静音 + **补零空洞**。对说话人嵌入而言：静音段不贡献音色但会稀释统计池化（CAM++/x-vector 类模型在 pooling 层按帧加权平均），空洞补的是**数字零**，等于往池化里掺确定性的假帧。`speech` 已经按 VAD 外包络 ±`pad_ms` 裁好，是「只有人声的那段」——正是嵌入模型训练时的输入分布。

代价：`speech` 依赖 VAD 边界质量。所以 §3 用 `vad_confidence` 与 `zeroed_in_speech_*` 反向给它兜底，并在 §11-C8 把「VAD 切边界伤声纹」列为需监测项。

### 3.2 去重与数据卫生（最容易埋雷的一处）

```
dedup_key = (tag, utterance_id, seq_first, seq_last, start_ms, end_ms)
```

同一句 bundle 被重复投喂（重放、多 sink、上层重试）必须**幂等**：同 key 第二次注册不新增样本。
第二层去重在向量上：`cos(new, existing) > dup_similarity`（默认 0.99）⇒ 视为同一句话的重复/近重复，拒收。
不做这层的后果：同一个人同一句话被切成两半分两次注册 ⇒ 库内该人样本数虚高、质心被一条真实 utterance 过度加权，
**并且注册样本同时是评测探针时构成数据泄漏**，指标会假性偏高（§11-C20 要求评测集与注册集按 `dedup_key` 互斥）。
这不是假想的坏味道：注册/测试之间的**音素重叠**本身就是 SdSV Challenge 2021 的正式评测维度（<3s 场景下它是最主要的
困难来源），而 WeSpeaker 官方打分口径特意写成 `cosine (sub mean of vox2_dev)` —— 说明**不减全局均值时 cosine 分数
整体偏高且互相相关**。也就是说「同一句切两半」在裸 cosine 下几乎必然骗过 0.99 之外的第二道判据。
另外 sherpa-onnx 的 `Add` 遇到重名只返回 `false`、一人一行，**不会**报告重复注册，去重必须在业务层做。

### 3.3 离线批量导入：没有实时 bundle 怎么办

在线唯一入口是 `UtteranceBundle`（§5.3：任何方法都不接受裸 `bytes`/`ndarray`，为的是让质量门控无法被绕过）。
但注册集、校准集、换模型重抽都是**离线**场景，手里只有 wav。这一条口子必须显式设计，否则一定有人为了省事
绕过门控直接塞音频。

做法：`bundle_from_wav_dir(dir)` 读 **segment `demo3 --utt-out` 的输出目录**重建 bundle —— 那里已经有
`<tag>/utt000001_416-10464_silence.wav`、同名 `_trim.wav`，以及一行一句的 `manifest.jsonl`
（`Utterance.manifest()` 的全部字段 + `files` 路径映射）。**以 `manifest.files` 为准取文件，不猜文件名。**

| bundle 字段 | 重建来源 |
|-------------|----------|
| `speech` 视图 | `files["trim"]`（segment 已按 VAD 边界 ± `pad_ms` 裁好，与 `to_bundle()` 的 `speech` 同区间） |
| `full` 视图 | `files["full"]` |
| `duration_ms / peak / mean / level` | 从 wav **实测**；与 manifest 里的 `mean_level_db` / `peak_level_db` 交叉校验，不一致就拒（说明 wav 被动过或裁剪口径变了） |
| `holes / zeroed_spans_ms / suspect_reasons` | 由 manifest 的 `holes`（正 = 补零）与 `parts[].status != clean` **推出**，口径与 `to_bundle()` 一致 |
| `close_reason / aborted / speech_span / vad_confidence` | manifest 原样 |
| `usable_for` | 用**当前** `VoiceprintConfig` 重算（阈值可能已变，不能沿用落盘时的判定） |
| `SourceRef.synthetic` | **置 `True`**：导入样本与在线样本可区分，`explain()` 与评测都能按此分桶 |

两个必须写死的例外：

1. `speech_spans_ms` 为空时，`_trim.wav` 走的是 `trim_to_speech()` 的**能量兜底**，而 `to_bundle()` 在这种情况下
   给的是**空视图**。重建必须照 `to_bundle` 的口径把 `speech` 置空（⇒ `no_speech` 拒），否则离线导入会绕过
   VAD 门控，塞进一堆没有语音的文件。
2. `keep_trimmed=False`（`--no-trim`）时没有 `speech` 来源 ⇒ 直接拒绝导入，**不允许**用 `full` 顶替（§3.1）。

这样三件事一并解决：注册集可以离线批量准备；换 embedding 模型时能从留存的 wav 重抽（= §5.4 的迁移路径，
不必新增 `keep_audio_dir` 概念，直接把 segment 的 `--utt-out` 目录当留存介质）；保留期由 segment 侧统一管理，
本模块不另存一份音频副本。

> **前置条件 P1（segment 侧 1 行改动，未落地）**：`UtteranceWriter.write()` 顺手把 `bundle.json()` 也写进
> manifest 行。现在能靠 `holes` + `parts[].status` 推出 suspect/zeroed 口径，但那是**第二次实现同一套判据**，
> 两处口径迟早会漂；写完 `bundle.json()` 后重建就退化成「读字段」，§11-C3 的可信度才立得住。

---

## 4. 数据结构（DTO）

```python
# com/robot/voiceprint/models.py（设计稿）

class TemplateState(Enum):
    DRAFT = "draft"          # 已 propose 未 commit，不参与者检索
    ACTIVE = "active"        # 正常参与
    DEGRADED = "degraded"    # 质量/数量不足或被 suspect 样本污染，参与但分数打折并告警
    STALE = "stale"          # 超过 stale_days 未见/未更新，默认不参与 identify（调用方传 include_states=(ACTIVE, STALE) 放行）
    RETIRED = "retired"      # 软删除：不检索、不导出，保留审计指向，purge 后物理删除

class Decision(Enum):
    KNOWN = "known"          # top1 ≥ threshold 且 margin 足够
    UNKNOWN = "unknown"      # top1 < threshold ⇒ 库里没这个人
    AMBIGUOUS = "ambiguous"  # top1 ≥ threshold 但 top1 − top2 < margin ⇒ 不敢说

class Verdict(str, Enum):    # 拒绝/降级原因，落日志可 grep
    OK = "ok"
    NO_SPEECH = "no_speech"          # speech 视图为空
    TOO_SHORT = "too_short"          # 低于 min_probe_ms / min_voiceprint_ms
    LOW_SNR = "low_snr"
    CLIPPED = "clipped"
    SUSPECT = "suspect"              # 命中 suspect 黑名单（空洞/补零/gap/clamped）
    TRUNCATED = "truncated"          # aborted 或 max_duration，不得用于注册
    DUPLICATE = "duplicate"          # dedup_key 或向量近重
    RATE = "sample_rate_mismatch"
    MODEL = "model_mismatch"
    CAPACITY = "capacity_exceeded"
    TTS_ECHO = "tts_echo"            # §11-C15：机器人自己的声音
```

```python
@dataclass(frozen=True)
class QualityReport:
    duration_ms: int
    snr_db: float                    # mean_level_db − noise_floor_db（来自 bundle）
    peak_level_db: float
    suspect_reasons: Tuple[str, ...]
    quality: float                   # 0..1，各子项加权（§7.4）
    blocked_by: Tuple[Verdict, ...]  # 空 ⇒ 可入库
    def ok(self) -> bool: ...
    def as_dict(self) -> dict: ...

@dataclass(frozen=True)
class SourceRef:
    """一条样本「从哪来」的**最小**记录：只存元信息，不存音频。"""
    tag: str
    utterance_id: int
    seq_first: int
    seq_last: int
    start_ms: int                    # 会话音频时钟
    end_ms: int
    speech_span_ms: Tuple[int, int]  # VAD 外包络（未加 pad_ms 的那对边界）
    sample_rate: int
    channels: int
    close_reason: Optional[str]
    bundle_json: str                 # bundle.json() 原文，字段扩充时不用改表
    synthetic: bool = False          # True = §3.3 离线重建的样本（手里只有 wav+manifest），评测与 explain 按此分桶
    via_pipeline: bool = False       # True = 走 §5.5 适配器来的，无 QualityReport ⇒ 只匹配、不注册/不 absorb

@dataclass(frozen=True)
class SpeakerSample:
    sample_id: str                   # "spk_xxx/0007"，稳定、可单独删除（split 用）
    speaker_id: str
    vector: np.ndarray               # float32, 已 L2 归一化
    model_id: str                    # 向量与模型版本强绑定（§5.4）
    source: SourceRef
    quality: float
    created_at: float                # UTC epoch 秒

@dataclass
class SpeakerTemplate:
    speaker_id: str                  # 主键，创建后永不变、不复用
    display_name: str                # 人类可读「小明」，可改；同名校验见 §6.4
    labels: FrozenSet[str]           # "关系:家人"、"权限:监护人"、"设备:客厅" —— 见 §6.4
    state: TemplateState
    model_id: str
    dim: int
    centroid: np.ndarray             # float32 归一化；由 samples 精确重算，不是增量近似
    sample_ids: Tuple[str, ...]
    n_samples: int
    quality_mean: float
    created_at: float
    updated_at: float                # 样本或标签变化才动
    last_seen_at: Optional[float]    # 最近一次 identify 命中（墙钟），供 aging
    enroll_notes: str                # 「读了哪句提示词」「在客厅录的」等，人工可查
    def as_dict(self) -> dict: ...

@dataclass(frozen=True)
class RankedHit:
    speaker_id: str
    display_name: str
    score: float                     # 原始 cosine
    norm_score: Optional[float]      # 归一化后（未启用归一化时 = None）
    labels: FrozenSet[str]
    n_samples: int
    state: TemplateState

@dataclass(frozen=True)
class IdentifyResult:
    decision: Decision
    ranked: Tuple[RankedHit, ...]    # 已按标签过滤后的 top_k
    threshold: float
    margin: float                    # top1 − top2（只有 1 个候选时 = top1 − 全库次优/0）
    probe_quality: QualityReport
    probe_source: SourceRef
    extract_ms: float
    search_ms: float
    model_id: str
    def best(self) -> Optional[RankedHit]: ...
    def speaker_id(self) -> Optional[str]: ...   # decision != KNOWN ⇒ None
    def explain(self) -> str: ...                # 一行讲清凭什么
    def as_dict(self) -> dict: ...               # 不含向量

@dataclass(frozen=True)
class VerifyResult:
    accept: bool
    speaker_id: str
    score: float
    threshold: float                 # 可以是该人的个性化阈值（§7.2）
    margin: float
    probe_quality: QualityReport
    def explain(self) -> str: ...
```

---

## 5. 接口定义

### 5.1 提取器（可插拔后端）

```python
# com/robot/voiceprint/interfaces/extractor.py

class IVoiceprintExtractor(ABC):
    """bundle 的语音视图 → L2 归一化的 float32 向量。"""

    @abstractmethod
    def model_id(self) -> str: ...           # 写进每条向量，参与兼容性校验
    @abstractmethod
    def dim(self) -> int: ...
    @abstractmethod
    def sample_rate(self) -> int: ...        # 声明"我要 16k"，不达标由调用方报错
    @abstractmethod
    def extract(self, view: AudioView) -> np.ndarray: ...
    def extract_batch(self, views: Sequence[AudioView]) -> List[np.ndarray]:
        ...                                  # 默认循环；onnx 后端可覆写成一次多批
```

实现清单（`impl/`）：

| 实现 | 依赖 | 模型与量级 | 用途 |
|------|------|-----------|------|
| `OnnxSpeakerExtractor` | `onnxruntime` 或 `sherpa_onnx.SpeakerEmbeddingExtractor` | 官方预导出 onnx（sherpa-onnx release tag `speaker-recongition-models`，36 个 asset）：**CAM++ 28.3–29.6 MB / 192 维 / 7.18M 参数 / 1.15 GFLOPs(≈2s 语音)**；ERes2Net-base 39.6 MB、V2 71.4 MB、large 116.1 MB、eres2net(200k) 220.6 MB；wespeaker resnet34 26.5 MB / 256 维 / **4.55 GFLOPs**；NeMo speakernet 23.4 MB、titanet-small 40.3 MB | **默认生产档**：免 torch、可离线、有 int8 变体与 arm64 APK，官方 README 明确支持 arm32/arm64/riscv64 / 树莓派 / RV1126。**代码与打包 Apache-2.0；权重许可必须回源模型卡逐条确认**（坑 1） |
| `FunasrSvExtractor` | `funasr` | 同族权重，走 `AutoModel(task="sv")` 系（`sv` / `sv-lasta` / `seacoil_sv` 才输出 `spk_embedding`） | 与仓库现有 `asr.py` 同一条加载路径，做 A/B 对齐 |
| `StubExtractor` | 纯标准库 | 按显式「说话人剧本」确定性造向量（同一剧本 → 同向量 + 可控扰动，不同剧本 → 近似正交；**模长可随传入视图时长线性放大**，用于复现长句主导质心这个坑） | **单测与自测**：§13 全部用例（含 TC28）在无模型环境跑通 |
| `CachedExtractor` | 包装器 | 按 `dedup_key` 缓存向量 | 注册时人工重看同一条不再重算 |

**选型上必须先清的三个坑**（直接影响能不能商用、分数能不能对得上）：

1. **许可（三层，一层一坑）**：
   * **权重层**（⚠ **本条已于 2026-11-19 订正**）：直读 ModelScope 模型 API，`damo`/`iic` 两个路径解析到**同一个** model_id
     `iic/speech_campplus_sv_zh-cn_16k-common`，其 `"License": "Apache License 2.0"`，且 README 全文**没有**任何
     「仅限研究用途 / 不可用于商业用途」的句子，`LicenseLink` 与 `LicenseName` 均为空。
     ⇒ 本文早先记录的「`damo/*` 权重页写研究用途」**不成立，已作废**（订正记录见文末存疑 12）。
     但**不能反过来当成「许可干净」**：页面给了 License 字段却没给可核对的正文（`LicenseLink` 空），而 FunASR 仓库另有
     `MODEL_LICENSE`（"Model Open Source License Agreement"，据核查无禁止商用条款、但要求署名与保留模型名、违约即终止）——
     **两处口径不一致，出厂时必须以随包附带的那一份文本为准**，法务确认前按更严的读。
   * **派生打包层**：sherpa-onnx 的 onnx 发布（release tag `speaker-recongition-models`）**代码与打包是 Apache-2.0，
     但不改变里面权重的许可**；导出页自身不写 license 段落（存疑 13）。
   * **数据集层（真正的坑）**：**WeSpeaker 官方 `docs/pretrained.md` 写明预训练模型许可跟随训练数据集**——
     VoxCeleb 版 → **CC-BY-4.0**（可商用、需署名）；**CN-Celeb 版跟随 CN-Celeb 协议**，而 CN-Celeb 官方页
     白纸黑字写 **"No commerical usage is permitted."**（原文如此拼写）——
     ⇒ **凡是「中文声纹」就顺手选 CN-Celeb 系权重的，法律上直接踩线**，这一条比 CAM++ 的字段问题危险得多。
   * 另：`pyannote/embedding` 需接受条款 + HF token（gated），HF gating 属合同层义务（不得转售、不得用于训练第三方模型）。
   所以落地前的动作是：**固定一个「权重 + 数据集」组合，把它的 License 正文与数据集协议一起存档**，
   而不是继续在博客口径的「可商用 / 不可商用」之间猜。
   另注：网传 `huggingface.co/3d-speaker/sv-cam++`「Apache-2.0 镜像」**不可作为依据**——2026-11-18 实测该 org 页 404，
   且 `sv-cam++` 本身是 GitHub 仓库里的**配方目录**名，不是 HF model id（见文末存疑 4）。
   **这条比精度重要，先于任何调参。**
2. **funasr 的 task 名**：新版 `task="dpda"` 路径只返回分段（`key/text/timestamps`），**不返回 embedding**；
   要向量必须走 `sv` 系 task 或 modelscope `pipeline(task='speaker-verification', output_emb=True)`。
   照抄网上「funasr dpda 取 spk_embedding」的片段会直接 `KeyError`。
3. **下载源**：PyPI `funasr==1.3.14` 内嵌的 `iic/…` asset 在 HF hub 上不存在，需
   `snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', revision='v2.0.1', model_hub='ms')` 走 ModelScope；
   板子离线运行要预置权重 + `disable_update=True`（仓库 `asr.py` 已是这个做法）。

> 后端只影响 `model_id`、耗时与**许可**，不影响任何判决口径与库结构；换后端 = 迁移（§5.4）。

**关于耗时，先把话说死**：3D-Speaker / sherpa-onnx / WeSpeaker / SpeechBrain / NeMo **均未发布** speaker-embedding
的 CPU 实时率或单条延迟基准，公开可查的只有 FLOPs（CAM++ 1.15 G @ 200 帧≈2s；wespeaker resnet34 4.55 G，
**贵 4 倍**）。所以板子上选模型的第一道筛子是 FLOPs 不是 EER，第二道才是 §12 的自测。
官方 EER 对照（3D-Speaker README 表，VoxCeleb1-O / CN-Celeb / 3D-Speaker）：
CAM++ 0.65 / 6.78 / 7.75；ERes2Net-base 0.84 / 6.69 / 7.21；ERes2NetV2 0.61 / 6.14 / 6.52；
ECAPA 0.86 / 8.01 / 8.87；ResNet34 1.05 / 6.92 / 7.29。中文家庭场景应优先看 CN-Celeb 与 3D-Speaker 两列。


### 5.2 存储

```python
class IVoiceprintStore(ABC):
    """模板与样本的持久化。实现要满足：读多写少、写原子、进程内线程安全。"""
    def meta(self) -> StoreMeta: ...                          # schema_version / model_id / dim / 阈值表版本
    def get(self, speaker_id: str) -> Optional[SpeakerTemplate]: ...
    def list(self, *, states=..., labels=..., label_match="all", limit=0) -> List[SpeakerTemplate]: ...
    def samples(self, speaker_id: str) -> List[SpeakerSample]: ...
    def matrix(self) -> Tuple[List[str], np.ndarray]: ...     # (speaker_ids, centroid 矩阵) 供检索；带内存缓存
    def put(self, tpl: SpeakerTemplate, samples: Sequence[SpeakerSample]) -> None: ...
    def delete(self, speaker_id: str, *, hard: bool) -> None: ...
    def append_audit(self, record: AuditRecord) -> None: ...
    def export(self, dest_dir: str, *, encrypt: bool = False) -> str: ...   # 返回包路径 + 校验和
    def import_from(self, pkg: str, *, on_conflict: "skip|merge|overwrite") -> ImportReport: ...
```

### 5.3 门面（产品只该碰这个）

```python
class VoiceprintService:
    # ---- 注册 ----
    def propose(self, bundle, *, speaker_id=None, name=None, labels=()) -> Draft:
        """只跑质量门控 + 提向量，落 draft 区，不进 gallery。可连续 propose 多句。"""
    def commit(self, draft_id: str) -> SpeakerTemplate: ...
    def enroll(self, bundles, *, name=None, labels=(), speaker_id=None,
               mode="create|append|replace", on_duplicate="reject|ignore") -> EnrollReceipt:
        """一步式（内部= propose N + commit）。bundles 是 List[UtteranceBundle]。"""
    # ---- 更新 ----
    def add_samples(self, speaker_id: str, bundles, *, auto_approve=True) -> EnrollReceipt: ...
    def remove_sample(self, sample_id: str) -> None: ...          # 拆分/清洗用，质心自动重算
    def rename(self, speaker_id: str, display_name: str) -> None: ...
    def set_state(self, speaker_id: str, state: TemplateState) -> None: ...
    def recompute(self, speaker_id: str) -> SpeakerTemplate: ...  # 纯重算质心/质量，不动样本
    def absorb(self, result: IdentifyResult, *, require_confirm=False) -> bool:
        """自适应更新：把一次识别命中吸收为样本。门槛见 §6.3。"""
    def migrate_model(self, new_extractor, *, audio_source=None) -> MigrationReport:
        """换 embedding 模型。默认无音频可重抽 ⇒ 报告"需重注册"；给了 audio_source 才真迁移（§5.4）。"""
    # ---- 删除 ----
    def retire(self, speaker_id: str, reason: str = "") -> None: ...
    def delete(self, speaker_id: str, *, purge_backups=False) -> DeletionReceipt: ...
    def withdraw_consent(self, speaker_id: str) -> DeletionReceipt:
        """撤回同意 = delete(purge_backups=True) + 轮换主密钥（§11-C23）。入口与注册同级（GDPR Art. 7(3)）。"""
    def forget_by_label(self, label: str, *, confirm_token: str) -> List[str]:
        """一键清除（GDPR 口径）：如 forget_by_label("主人:张三")。confirm_token 必填，防误触。"""
    # ---- 查找 ----
    def identify(self, bundle, *, top_k=5, labels=(), label_match="all",
                 include_states=(ACTIVE,), allow_short=False) -> IdentifyResult: ...
    def verify(self, bundle, speaker_id: str) -> VerifyResult: ...
    def search(self, vector, *, top_k=10, threshold=0.0, labels=()) -> List[RankedHit]:
        """裸向量检索：给 speaker_pipeline 的 ISpeakerMatcher 用（§5.5）。"""
    def who_is(self, speaker_id: str) -> SpeakerProfile: ...      # 模板 + 标签 + 统计 + 最近命中
    # ---- 标签 ----
    def add_labels(self, speaker_id: str, labels) -> None: ...
    def remove_labels(self, speaker_id: str, labels) -> None: ...
    def rename_label(self, old: str, new: str) -> int: ...        # 返回受影响模板数
    def labels(self) -> List[LabelStat]: ...                      # 每个标签的人数、命名空间树
    def find_by_label(self, label: str, *, match="all", any_of=()) -> List[SpeakerTemplate]: ...
    # ---- 维护 / 评测 ----
    def audit_quality(self) -> List[TemplateIssue]: ...           # 样本少/分歧大/过期/标签冲突
    def merge(self, keep_id: str, drop_id: str) -> SpeakerTemplate: ...   # 同一人被注册成两个
    def split(self, speaker_id: str, sample_ids, new_name: str) -> SpeakerTemplate: ...
    def stats(self) -> StoreStats: ...
    def evaluate(self, probes) -> EvalReport: ...                 # EER/FAR/FRR/acc（§13.3）
```

所有写操作签名都带 `actor`（`"cli" / "app" / "auto"`）与 `reason`，落 `audit.jsonl`。
**没有**任何方法接受 `bytes`/`ndarray` 音频 —— 音频入口只有 `UtteranceBundle`，逼调用方走质量门控。

### 5.4 模型版本绑定（不能省的一条）

每条向量存 `model_id`；`StoreMeta` 存当前 `model_id + dim + feature_config`。

* `identify`/`verify` 时 probe 的 `model_id` 与库不一致 ⇒ `ModelMismatchError`，**绝不跨版本算 cosine**（不同模型的向量空间毫无可比性）。
* 同一库内允许存在历史 `model_id` 的样本，但它们不参与检索（`state=RETIRED` 的向量段），迁移完再清。
* 迁移的真实代价：本模块默认**不存原始音频**，所以换模型 = 全员重注册。唯一的省法是把 segment 侧
  `demo3 --utt-out` 落下的整句 wav 当留存介质，用 §3.3 的 `bundle_from_wav_dir()` 离线重抽
  （配置项 `utt_archive_dir`，默认关闭）。**这条口子一开，音频的保留期与删除级联就同时管住了两个模块**，
  所以它是产品决定（§14-Q4），不是技术参数。
* 上游小改（P1，§3.3）没做之前，`utt_archive_dir` 的重建口径靠 `holes`/`status` 推导，视为**过渡实现**。

### 5.5 适配 `speaker_pipeline_design.md`

```python
class MatcherAdapter(ISpeakerMatcher):      # speaker_pipeline_design.md §4.5 的接口
    def register(self, speaker_id, embedding): ...     # 转 add_samples（要求已有模板）
    def match(self, embedding, top_k=1): ...           # 转 search()
    def is_known(self, embedding, threshold=...): ...  # 转 search + 阈值
    def match_batch(self, embeddings): ...             # 转 search × M（可并行）

class PipelineVoiceprintAdapter(IVoiceprintExtractor):  # speaker_pipeline_design.md §4.4 的同名接口
    def extract(self, segment: AudioSegment) -> VoiceprintEmbedding:
        ...   # AudioSegment → AudioView（无 VAD 元信息，speech=full）→ 本模块 extractor → 包成 VoiceprintEmbedding
    def feature_dim(self) -> int: ...                   # 转发本模块 extractor.dim()
```

> **同名不等于同一件事**：那边的 `IVoiceprintExtractor` 输入是流水线内部的 `AudioSegment`（波束后的裸帧），
> 输出是 `VoiceprintEmbedding`；本模块的 `IVoiceprintExtractor`（§5.1）输入只能是 `AudioView`，
> 且**必经 §3 的质量门控**。所以两边各自保留 ABC，靠上面这个适配器相连——本模块绝不反向依赖 pipeline 的数据结构。
> 走这条适配器的向量**没有** `QualityReport`（pipeline 那边没有 VAD/holes 上下文），因此只参与匹配、
> **不进入注册与 absorb**（`SourceRef.via_pipeline=True`，§4），否则门控就被绕过了一条口子。

反过来，本模块的 `identify(bundle)` 不做任何分离，只吃已经分好的一句话——职责单向。

---

## 6. 关键流程

### 6.1 注册（两步式，产品要的是这个）

```
引导：请说 3 句（覆盖 高/正常/小声），每句必须 usable_for.voiceprint == ok
  │
  ├ propose(bundle) → QualityReport → 向量 → Draft 区（不进 gallery）
  │        拒：TOO_SHORT / LOW_SNR / SUSPECT / TRUNCATED / DUPLICATE / TTS_ECHO
  ├ （人看：这三句的 quality 分、SNR、时长）
  └ commit(draft_id) → ACTIVE 模板 + audit.jsonl 记录 + 标签写入
```

为什么必须两步：**注册是唯一会污染库的入口**。一步式（直接 `enroll`）留给脚本与批量导入，但同一份质量门控照跑，只是少了人工看一眼的机会。

硬规则：

1. `close_reason == "max_duration"` 或 `aborted` 的 bundle **不得用于注册**（半句拉偏质心）。
2. 达到 `min_samples_per_speaker`（默认 3）才转 `ACTIVE`；不足时 `DEGRADED`，参与检索但 `explain()` 里明说。
3. `n_samples > max_samples_per_speaker`（默认 32）后拒绝再追加，要求人工挑最像质心的留下——防止单模板样本无限增长拖慢检索、也防止某一时段的录音主导模板（§11-C11）。

### 6.2 识别（在线，消费线程内）

```
bundle ──► usable_for["voiceprint"] 门控 ──► speech.to_numpy() ──► extractor.extract
                                                                        │ L2 归一化
                                                                       ▼
                              search(centroid 矩阵) ──► top-k ──► 阈值 + margin 判决
                                                                       │
                                                    KNOWN ──► last_seen_at 更新（异步）
                                                    UNKNOWN ──► 累计 unknown 计数（§11-C9 提示注册）
                                                    AMBIGUOUS ──► 上层不认人，只回内容
```

不阻塞对话：整条链路目标 `≤ 60 ms`（§12），放在 segment 的消费线程里同步跑即可；若上板实测超预算，改为「先回内容、异步补身份」的两段式回调（§14-Q5）。

### 6.3 更新 / 自适应吸收（危险操作，默认半自动）

`add_samples`（人工给已知的人补录音）是安全的。危险的是**自动吸收**：把 identify 命中直接并入模板，一旦认错一次，模板就开始向陌生人漂移，且错误自我强化。所以 `absorb()` 的门槛**高于** `identify`：

```
score ≥ absorb_threshold（默认 = identify_threshold + absorb_margin_delta，0.05 量级）
且 probe.quality ≥ absorb_min_quality（默认 0.7）
且 n_samples < max_samples_per_speaker
且 该 probe 与模板现有样本的 max cos < dup_similarity（防同一句反复吸收）
且 距上次 absorb ≥ absorb_cooldown_s（默认 3600）
```

任一不满足 ⇒ 只记 `pending_absorb` 候选，等人确认。质心一律**由样本精确重算**，不做 `c ← αc + (1−α)v` 的指数近似：`n ≤ 32, dim = 192` 时重算是微秒级，而指数近似会让早期错误永远残留（样本数少时 α 权重过大）。

**求和顺序有坑，照抄参考实现会中招**：sherpa-onnx 的 `SpeakerEmbeddingManager` 单条 `Add` 路径会显式逐行 normalize，
但**多样本合并路径是「先求和再 normalize」（注释写明不必除 N）**，没有逐条归一化。
因为 embedding 的**模长与语音时长/质量相关**（quality-aware calibration 正是把时长当质量因子），
一条 8s 的样本会以数倍权重压过一条 1.5s 的样本，质心被最长的那句主导。本模块的口径写死为
**`逐条 normalize → 求均值 → 再 normalize`**（与 AS-Norm 原文的模板平均顺序一致），并要求 `SpeakerSample` 落盘时
存的是**已归一化**向量；`recompute()` 里加一条断言 `abs(norm(v) - 1) < 1e-3`，不满足直接报错而不是静默修正。

改名 / 改标签（`rename` / `add_labels` / `remove_labels` / `rename_label`）不动向量，单独审计。

### 6.4 删除

| 动作 | 语义 | 向量 | 备份 |
|------|------|------|------|
| `set_state(RETIRED)` | 软删除：立刻不检索、不导出、不出现在 `list()` | 保留（可撤销） | 保留 |
| `delete(speaker_id)` | 硬删除该人：从矩阵剔除并重写、样本文件删、`audit.jsonl` 只留**无向量**的动作记录 | 立即物理删除 | 默认保留到 `backup_retention_days`，过期或被 `purge_backups=True` 清掉 |
| `remove_sample(sample_id)` | 删一条样本（清洗/拆分） | 该条剔除，质心重算 | 同模板 |
| `forget_by_label(label)` | 批量清除某人名下**全部**（含同名不同 id） | 同上 | 同上 + 强制提示检查备份 |
| `withdraw_consent(id)` | 撤回同意：等价 `delete(..., purge_backups=True)`，并**轮换主密钥** | 旧密钥解不出 ⇒ 残留密文不可用（crypto-erasure） | 备份一并失效 |

删除必须是**级联且可验证的**：`delete()` 之后 `who_is()` 报 `SpeakerNotFoundError`、`search()` 不再命中、`matrix()` 行数 −1、`export()` 不含此人、`find_by_label()` 无残留。§13.2 有专门用例锁这五条。
审计记录**不得含**任何可还原声纹的信息（不存向量、不存 bundle `full` 的 pcm、不存音频路径之外的定位信息）——否则删除只是删数据、留了副本。

### 6.5 标签

* 形态：`namespace:value` 字符串集合，无命名空间即 `tag:*` 之外的裸标签。内置保留命名空间：
  `关系:`（家人/朋友/同事）、`权限:`（监护人/普通/访客，**影响下游能否执行动作**）、`设备:`/`环境:`（客厅/远场，用于信道归一化 §7.3）、`主人:`（数据主体标识，`forget_by_label` 的键）。
* `label_match="all"|"any"`；倒排索引 `label → {speaker_id}` 常驻内存，写操作同步更新，`list()` 走索引而不是全表过滤。
* `display_name` 与标签是两件事：name 唯一性只做**大小写/空白归一后**的冲突检查（重名允许但 `rename` 返回 `warn="name_conflict"`），因为声纹的主键是 `speaker_id`，不是名字。
* 一致性辅助（§11-C17）：同一 `关系:家人` 下若两个人向量 cos > 0.85，`audit_quality()` 报「可能是同一个人注册了两次」。

---

## 7. 判决与阈值

### 7.1 打分

`score = dot(v_probe, centroid)`（两边都已 L2 归一化）。默认用质心而非「逐样本 max」：质心 O(1) 存储、O(N·dim) 计算、对样本量不敏感；逐样本 max 精度略高但 O(N·M·dim)。库小（人数 < 200）时**两者都可负担**，故 `scoring="centroid|max_sample|blend"` 做成配置，默认 `centroid`，`blend = 0.7·centroid + 0.3·max_sample` 留给实测显示分不清的场景。

**明确不引入 PLDA**：PLDA 需要数千 labeled utterances 估计类内/类间协方差，几百人 × 几条的库根本训不出；
用别人训好的 PLDA 又与本库分数分布失配，反而破坏阈值可解释性（x-vector 原文用 Gaussian 对齐后等分超平面、
NIST 也建议小库用 cosine 打分）。要涨精度就换 ERes2Net2907 这种**更强后端**或换打分方式，而不是上 PLDA。
`max_sample` 检索的精度/存储代价对照见 §12。

### 7.2 阈值怎么来（不接受拍脑袋数字）

```
1. 采集校准集：≥ 20 人 × 每人 ≥ 5 句 + 尽量覆盖不同设备/距离/时段
2. 跑 identify，收集：同人异句分数分布 P_same；异人分数分布 P_diff（全对）
3. 在 ROC 上按业务代价选点：
   - 免打扰唤醒门（宁可认错人也不要不理人）→ 偏向低 FRR
   - 个性化内容/权限判定（宁可不理人也不要认错）→ 偏向低 FAR，甚至只允许 AMBIGUOUS→拒
4. 落阈值表：per model_id（换模型必重标）、per state（新注册模板样本少 → 阈值抬高）、
   可选 per speaker（样本多的老模板可以更松）
```

**阈值不可跨系统移植**，这条比上一条更重要，实测口径已经乱到必须写进文档：

* 3D-Speaker 文档给出的 **cosine 尺度** SV 阈值量级是 **0.40~0.55**（中文说话人、CAM++/ERes2Net）。
* ModelScope `pipeline(task='speaker-verification')` 的 `threshold: 0.31 / 0.365` 是**内置后处理之后**的分数
  （`score_postprocess_path='damo/speech_campplus_sv_transform_er/score_sensevo_0.83.ckpt'`），**不是裸 cosine**；
  网上大量 CAM++ 阈值教程抄的是这个数。
* ERes2Net 在 CN-Celeb 上的 minDCF 阈值写的是 0.14，同属该页面自己的 score 体系。
* 论文侧还有一条容易被忽略的约束：**未做自适应归化时，EER 阈值不能直接当 FAR 阈值用**
  （DFN 会议 16k 场景：非归化 EER 23.8% → 归化后 8.7%）。
* **两个能查到的官方数字**：sherpa-onnx 官方示例（Python / Rust 一致）的 `--threshold` 默认 **0.6**，
  打分方式恰好就是本模块采用的「逐条 normalize + 全矩阵点积取 max + 低于阈值判 unknown」，
  所以它是最贴近的可比锚点；3D-Speaker 的 `diar.yaml` 里 `mer_cos: 0.8` 是**说话人日志聚簇合并**阈值，
  用途不同，**不要**拿来做 1:N 判决。
* **比较范围也要写清**：公开 EER 表多数不是裸 cosine——WeSpeaker 明确写 `cosine (sub mean of vox2_dev)`，
  再叠加 AS-norm / QMF 还有额外收益（官方表：CN-Celeb ResNet34 EER 7.134→6.119、minDCF 0.408→0.354；
  vox1-O 0.867→0.659）。本模块**不减全局均值、不做归化**，所以别人的阈值和 EER 都只能当量级参考。

所以 `identify_threshold = 0.45` 只是**占位值**，而且按上面两个锚点看它**偏松**（0.45 < 官方示例 0.6）——
召回优先的场景可以将就，权限类场景必须往上调并靠 §13.3 的自标定重选。
阈值表随库版本化（`StoreMeta.thresholds_version`），改阈值 = 一次可回滚的迁移，不是改常量。


### 7.3 归一化（v2）

跨信道/跨会话分数不齐时上 T-norm/Z-norm：用 cohort（库内其他人）的分数把 raw cosine 做均值/方差标准化。实现前提是有 ≥ 数十人的库；人少时反而不稳，所以 v1 明确**不做**，只在 `norm_score` 字段留位、`explain()` 打印 `(raw, norm=None)`。

### 7.4 质量分

`quality = w1·时长分 + w2·SNR 分 + w3·完整性分 + w4·VAD 置信分`，各子项归一到 0..1，权重进配置表。用途三处：注册准入（`≥ min_quality_enroll`）、自适应吸收门槛（§6.3）、检索时的**证据**（`explain()` 里写 `q=0.62 (snr 9.1dB, dur 1.3s)`，不直接改分数——「用质量分加权分数」会让阈值失去物理含义，明确列为**不做**，除非校准集实测证明有效）。

---

## 8. 参数表（集中配置，单位统一 ms / dB / 秒）

```python
# com/robot/voiceprint/config.py（设计稿）

@dataclass
class VoiceprintConfig:
    # ---- 输入对齐（必须与 SegmentConfig 一致，validate() 里硬校验）----
    sample_rate: int = 16000
    channels: int = 1
    view: str = "speech"                # 音频视图，只允许 speech（§3.1）
    pad_ms: int = 40                    # 传给 to_bundle()，与 segment 默认一致

    # ---- 质量门控 ----
    min_probe_ms: int = 600             # 低于此不识别（比 bundle 的 min_voiceprint_ms=1000 宽一档）
    min_voiceprint_ms: int = 1000       # 注册下限：直接沿用 to_bundle 的判定口径
    min_enroll_ms: int = 3000           # 注册下限。依据三条：① 3D-Speaker 官方训练 config `wav_len: 3.0`
                                        #   （模型就工作在 3s 尺度上）；② sherpa-onnx 官方示例丢弃 <0.5s 段、
                                        #   `min_speech_duration=0.25`（下限侧的硬证据只有这么低）；
                                        #   ③ 短语音实验 5s→0.5s 时 EER 1.53%→9.27%。探针可以短，模板不能短。
    min_snr_db: float = 8.0
    max_peak_db: float = -1.0           # 高于此 ⇒ 削波嫌疑 CLIPPED
    vad_trust_ms: int = 60              # VAD 置信低时两端再缩这么多
    suspect_blocklist: Tuple[str, ...] = ("hole_", "zeroed_in_speech_", "clamped", "gap")
    min_quality_enroll: float = 0.55
    min_quality_probe: float = 0.35

    # ---- 注册 ----
    min_samples_per_speaker: int = 3
    max_samples_per_speaker: int = 32
    dup_similarity: float = 0.99
    allow_short_enroll: bool = False
    draft_ttl_s: int = 900              # propose 后 15 分钟不 commit 自动清理

    # ---- 判决 ----
    scoring: str = "centroid"           # centroid | max_sample | blend
    blend_weight: float = 0.7
    identify_threshold: float = 0.45    # ⚠ 未标定占位：文档口径 0.40~0.55，sherpa-onnx 示例默认 0.6（更严），见 §7.2
    ambiguous_margin: float = 0.05      # top1 − top2 < 此值 ⇒ AMBIGUOUS
    verify_threshold_delta: float = 0.03    # 1:1 比 1:N 严（候选少但代价高）
    top_k: int = 5
    include_stale_by_default: bool = False
    stale_days: int = 90

    # ---- 自适应更新 ----
    absorb_margin_delta: float = 0.05
    absorb_min_quality: float = 0.70
    absorb_cooldown_s: int = 3600
    absorb_needs_confirm: bool = True   # 产品默认半自动（§6.3）

    # ---- 库容量与延迟 ----
    max_speakers: int = 2000
    identify_budget_ms: float = 60.0
    extract_threads: int = 1

    # ---- 存储 / 隐私 ----
    root_dir: str = "var/voiceprint"
    backup_dir: str = ""                # 空 = 不做历史备份
    backup_retention_days: int = 30
    audit_retention_days: int = 365
    utt_archive_dir: str = ""           # 指向 segment 的 --utt-out 目录；默认关闭（§3.3 / §5.4）
    utt_archive_days: int = 7           # 开了就必须有过期，删除级联覆盖到这里
    encrypt_export: bool = False
    encrypt_templates: bool = False     # §11-C23；有 `权限:` 标签时必须 True（validate 里硬校验）
    unknown_prompt_after: int = 5       # 连续 N 次 unknown 提示注册（§11-C9）

    def validate(self) -> None: ...
```

**容量与延迟约束（必须作为断言 / CI 检查存在）**

```
(a) 输入对齐      sample_rate == SegmentConfig.sample_rate 且 channels == 1
(b) 注册门槛      min_enroll_ms ≥ min_voiceprint_ms ≥ min_probe_ms
(c) 判决单调      identify_threshold + absorb_margin_delta < 1.0
                  ambiguous_margin > 0 且 identify_threshold + ambiguous_margin ≤ 1.0
(d) 内存上限      max_speakers × (max_samples_per_speaker + 1) × dim × 4 B
                  = 2000 × 33 × 192 × 4 ≈ 51 MB   ← 超板子预算就调小
                  max_samples_per_speaker（→8 时约 14 MB）或 max_speakers
(e) 延迟预算      extract_ms + search_ms(max_speakers, scoring) + decide_ms ≤ identify_budget_ms
                  实测（x86 / numpy 单核，见 §12 表）：2000 人质心 0.10 ms、2000×32 逐样本 2.97 ms
                  ⇒ 该不等式几乎完全由 extract_ms 决定，模型耗时必须**上板实测**后才敢开识别
(f) 去重可判定    0.9 < dup_similarity < 1.0
(g) 安全下限      模板中出现保留标签 `权限:*` ⇒ encrypt_templates 必须为 True，且
                  utt_archive_dir 为空（有权限判定就不许留原始语音档案）
```

(d) 是**唯一会让人犹豫的地方**：只存质心时 2000 人只要 `2000 × 192 × 4 = 1.5 MB`；
连逐样本一起存（为了 split/重算/max_sample）才到 51 MB 量级。板子内存不够时把
`max_samples_per_speaker` 降到 8（≈14 MB）或把逐样本向量落盘、只缓存质心。

---

## 9. 落库结构（对齐仓库现有命名与写法）

```
com/robot/voiceprint/
├── __init__.py           # re-export：VoiceprintService/Config/DTO/Extractor 工厂
├── models.py             # §4 DTO + 枚举（纯标准库）
├── config.py             # §8 参数 + validate()（对齐 SegmentConfig 风格，配错启动即 ConfigError）
├── quality.py            # bundle → QualityReport（§3 的门控表在这里，唯一一处）
├── scorer.py             # 归一化、质心、打分、阈值/margin 判决（纯 numpy/数组，无模型依赖）
├── store.py              # §5.2 的 JSON + npz 实现（原子写，参照 scheduler/disk_store.py）
├── tags.py               # 命名空间解析 + 倒排索引 + 标签校验
├── service.py            # §5.3 门面：生命周期编排 + 审计
├── interfaces/extractor.py
├── impl/
│   ├── extractor_funasr.py     # 惰性 import funasr
│   ├── extractor_onnx.py       # 惰性 import sherpa_onnx
│   ├── extractor_stub.py       # 确定性伪向量，CI 用
│   ├── matcher_adapter.py      # speaker_pipeline §4.5 ISpeakerMatcher 的实现
│   └── pipeline_adapter.py     # speaker_pipeline §4.4 IVoiceprintExtractor 的实现（§5.5）
├── demo.py               # CLI：enroll / identify / who / list / tags / remove / export / eval
└── test_voiceprint.py    # §13 用例（stub 后端，纯标准库 + numpy）

var/voiceprint/           # 运行期数据（.gitignore）
├── meta.json             # schema_version、model_id、dim、thresholds_version、created_at
├── speakers.json         # {speaker_id: 模板元信息 + labels + state + 时间戳}   ← 轻量，人可读
├── embeddings.npz        # centroids (n×dim, f4)、sample_index、sample_vectors(f4)
├── labels.json           # 倒排索引 {label: [speaker_id]}（启动重建，落盘只为快）
├── samples/<speaker_id>.jsonl   # 每条样本的 SourceRef + QualityReport（**无向量无音频**）
├── drafts/<draft_id>.json       # propose 未 commit 的临时区
└── audit.jsonl           # 追加式：谁在什么时候对谁做了什么，字段见 §10.3
```

写法沿用仓库既有约定（`scheduler/disk_store.py`）：**临时文件 + `os.replace` 原子落盘**、
`threading.RLock` 保护写、读走内存缓存 + `index` 文件。`np.savez` 到临时文件再 `os.replace`，
避免半截 npz。SQLite 是另一个合理选择（事务/并发/查询），但仓库现在没有 sql 先例、
`sqlite3` 又是标准库，所以 v1 用 JSON+npz 保持同构，**把「换 SQLite」记在 §14-Q3**。

---

## 10. 隐私与合规

### 10.1 定性（以及两个必须讲清的限定）

声纹是**生物特征数据**：不可更换、可跨设备关联、本人通常无法察觉被记录。

* **GDPR**：Art. 4(14)「genetic data」条目下相邻定义的 biometric data 指「具体的技术处理后的个人生物特征数据，
  且能**验证或确定**该自然人的身份」；Art. 9(1) 把「biometric data for the purpose of uniquely identifying
  a natural person」列为特殊类别个人数据，**原则禁止处理**，同意须符合 Art. 4(11) 的定义（自由、具体、知情、明确）
  并按 Art. 9(2)(a) 走「明确同意」；配套权利包括知情（13/14）、访问（15）、更正（16）、**删除/被遗忘（17）**、
  限制处理（18）、**可携带（20）**、反对（21）；同意还必须**随时可撤回、撤回与给予同样容易**（Art. 7(3)；明确同意的口径见 Recital 42）。
  用于唯一识别的声纹落入特殊类别，故适用 Art. 9。
* **一个限定，别当成免死金牌**：EDPB 立场是「只有当语音被用于**确定/验证**身份时才算生物特征数据，
  只有语音本身时可能不算」。但本模块**正是**用于唯一识别 ⇒ 没有回旋余地；且未加密语音即便被认定不属
  特殊类别，仍属个人数据，GDPR 其余义务照旧。
* **国内口径同样吃严规则**：《个人信息保护法》第 28 条把「生物识别」列入**敏感个人信息**，
  第 29 条要求**单独同意**（不得与其他同意打包），第 30 条要求告知必要性与影响。
  机器人如果在国内落地，走的是这条而不是 GDPR，但工程落点几乎一致。
* **模板 ≠ 可安全公开的字符串**：声纹模板不可逆推语音，但**可以**被直接拿去做说话人识别——
  「不可还原为语音」不等于安全。NIST SP 800-63-3 §5.2.3.1 明确要求生物特征模板在不使用受保密保护的
  密钥时**不得作为认证器使用**、存储与传输必须**加密或哈希**。所以「权限:监护人」这类判定
  （§14-Q1）在本模块里必须叠加第二因子。

因此本模块的默认姿态是：**本地存、默认不存音频、可查可删、删得干净、模板按密钥保护**。

### 10.2 设计里已经体现的约束

| 要求 | 落点 |
|------|------|
| 告知 + 明确同意 | `enroll` 必须带 `consent=True`（CLI 交互确认 / App 传标志），未同意 ⇒ 拒。记录在 `audit.jsonl` 的 `consent` 字段。国内口径要求**单独同意**（PIPL 第 29 条），故不能塞进「我已阅读用户协议」这种打包勾选 |
| 撤回同样容易 | 撤回入口与注册入口同级；`withdraw_consent(speaker_id)` = `delete(purge_backups=True)` 的语义别名，一条命令走完（GDPR Art. 7(3)） |
| 数据最小化 | 默认**不落原始音频**（`utt_archive_dir` 关闭）；样本记录只有 `SourceRef` 元信息 |
| 存储限制 | `backup_retention_days` / `audit_retention_days` / `utt_archive_days` 三个过期参数 + 启动时清扫 |
| 删除权 | `delete` / `forget_by_label` 的级联可验证语义（§6.4），含备份 |
| 可携带 | `export()` 出 jsonl + npz + meta（可加密、带校验和），`import_from()` 带冲突策略 |
| 完整性与保密性 | `encrypt_export` + `meta.schema_version` 校验 + 导出包 sha256；明文音频永不出目录；模板本体加密 + 密钥轮换实现「删除可自证」（§11-C23，GDPR Art. 32(1)(a) 明确要求假名化与加密，NIST SP 800-63-3 §5.2.3.1 要求模板不得以明文当认证器） |
| 事前评估（DPIA） | 每人一条声纹 = 特殊类别数据的系统性处理，GDPR Art. 35(3)(b) 要求**事前**做 DPIA。工程侧要能随时回答三个问题：存了什么（`samples/*.jsonl` + `speakers.json`）、给过谁（`audit.jsonl`）、删没删（§6.4 的五条断言）。这三个回答不出来，DPIA 就写不了 |
| 儿童数据 | 标签 `关系:家人` 不区分成年与否，但 PIPL 第 31 条要求**不满 14 周岁须监护人同意并制定专门处理规则**；§11-C19 要求产品侧对儿童声纹单独同意 + 更短保留期，模型对儿童效果也需单独校准（本文只把它列为风险，不承诺效果） |
| 业界先例（抄作业的地方） | Google Voice Match：停用即「从设备删模型 + 从账号删录音」，设备侧最长 24 h 生效，不活跃模型 12 个月自动过期，并明确提示「**在某些地区语音模型可能被视为生物特征数据**」；Amazon Alexa：云端不留存录音则关联 voice ID **一并删除、功能直接失效**（3/18 个月自动删除可配）；Apple Siri：尽量设备端 + 随机标识符。⇒ 三条共识本模块都对齐：本地优先（O7）、删除必然导致功能降级（要在 UI 里讲清楚，不能让用户以为只是「关个开关」）、给撤回一个明确的生效时限 |

### 10.3 审计记录（`audit.jsonl`，一行一动作）

```json
{"at": 1763400000, "actor": "cli", "op": "enroll.commit", "speaker_id": "spk_7f3a",
 "name": "小明", "labels": ["关系:朋友"], "n_samples": 3, "quality_mean": 0.81,
 "model_id": "campplus-192-v1", "consent": true, "source": {"tag": "mix_mean",
 "utterance_id": 42, "start_ms": 10752, "end_ms": 12768}, "reason": ""}
```

不含向量、不含音频、不含 pcm 摘要。识别（`identify`）默认**不**写这里，而是写运行日志
（`voiceprint_events.jsonl`，含分数与判决，用于线上指标与误识复核），保留期独立可配。

---

## 11. 还需要什么能力（逐条：问题 → 影响 → 方案 → 分期）

这是 §3–§10 之外还要补的能力清单。「一期」列为最小可用集（缺一条就会埋雷），其余标注引入时机。

| # | 能力 | 问题 / 影响 | 方案要点 | 期 |
|---|------|-------------|----------|-----|
| C1 | **拒绝/未注册判定** | 只给 top1 会把陌生人硬塞成最像的熟人 | `Decision.UNKNOWN` + 阈值 + 校准集；`verify` 与 `identify` 分开阈值 | 一期 |
| C2 | **模糊保护** | 家里两个人音色相近，认错比拒识代价大 | `AMBIGUOUS` + `ambiguous_margin`；上层降级成「不认人只回内容」 | 一期 |
| C3 | **幂等与去重** | 同句重复注册 ⇒ 质心偏、评测泄漏 | `dedup_key` + `dup_similarity`（§3.2） | 一期 |
| C4 | **质量门控** | 远场小声 / 削波 / 含补零的句子进库，模板被永久污染 | `QualityReport` + 准入表（§3）+ `min_quality_enroll` | 一期 |
| C5 | **导出 / 导入 / 备份 / 校验** | 换机、灰度回滚、审计取证 | `export()/import_from()`，sha256，`on_conflict` | 一期 |
| C6 | **模板可撤销**（快照） | 一次误注册或误吸收毁掉一个人 | `n_samples` 前的 `pre_state` 快照 + `rollback(speaker_id, to_ts)` | 一期 |
| C7 | **merge / split** | 同一人注册成两个人；两个人被并成一个 | `merge(keep, drop)` 合样本重算；`split(sample_ids)` 剥出新模板 | 一期 |
| C8 | **VAD 边界对声纹的伤害监测** | `speech` 视图全靠 VAD；切进辅音/漏掉尾巴会拉低分数 | `quality.vad_trust` + 事件日志里带 `speech_span` 与 `close_reason`；指标发现「注册句普遍短」就调 `pad_ms` | [一期]仅观测 |
| C9 | **unknown → 引导注册** | 新产品期库里没人，识别永远 unknown，用户不知道要注册 | 连续 `unknown_prompt_after` 次 ⇒ 发 `RegisterPrompt` 事件给对话层；unknown 向量做短期聚类，同一「疑似新人」出现 N 次才提示（避免一次噪声就弹窗） | 二期 |
| C10 | **信道/采样率链路一致性** | 44.1k/48k → 16k 只保留 0–8 kHz，**上采样不会恢复高频**；48k 注册 + 44.1k 使用的跨采样率对，在 VoxCeleb1-O 上使 baseline EER 从 2.31% 升到 3.22%，**重采样方式的影响比编解码更大**；**同一模型**跨设备/跨距离/跨方言的官方实测劣化更大：CAM++ 在 3D-Speaker 集上 Cross-Device 7.75% → Cross-Distance 11.29% → Cross-Dialect 13.44%（ECAPA 8.87/12.26/14.53），**距离一项就吃掉 3.5 个百分点** | 注册与识别尽量同链路；标签 `设备:`/`环境:` 参与 cohort 与分桶指标；采样率不一致时按 §3 直接拒（不静默重采样）；训练侧靠 `aug_prob: 0.8` 的加噪+混响增强，但那只是缓解不是解决 | 二期 |
| C11 | **老化与漂移监测** | 三个月后老模板分数系统性下降，误拒增多 | `last_seen_at` + `stale_days` + `audit_quality()` 报「样本过旧 / 内部分歧大 / 均值近期分数低」；提示重注册而非自动改 | 二期 |
| C12 | **受控自适应（absorb）** | 不吸收 ⇒ 模板永远停在注册当天；乱吸收 ⇒ 投毒与漂移 | §6.3 五重门槛 + `absorb_needs_confirm` 默认 True + 全量审计 | [二期]（默认关闭） |
| C13 | **阈值治理与回归集** | 换模型/换阈值后没人知道有没有变差 | `evaluate(probes)` 出 EER/FAR/FRR/acc/混淆；阈值表版本化；CI 用固定回归集跑指标不回退 | [一期]脚本，[二期]进 CI |
| C14 | **反欺骗 / 活体** | 录音回放、TTS 拟声可冒充注册人；本模块 v1 **拦不住** | 明确写进风险；留 `liveness` 字段与 `IVoiceprintAntiSpoof` 接口位；权限类判定（`权限:监护人`）叠加 PIN/唤醒等**第二因子**，不把声纹当唯一凭据 | [三期]，产品必须先知情 |
| C15 | **自听 / TTS 隔离** | 机器人自己说话（或音箱外放）被当新用户注册进去 | 复用 segment 的门控（`gate_mode`/`ignore_during_tts`）；`enroll` 时 `AEC` 未确认 ⇒ 拒并要求 `gate.tts_playing==False`；事件里带 `Verdict.TTS_ECHO` | 一期 |
| C16 | **配额与检索上限** | 单人塞 500 条样本拖慢全场、撑爆内存 | `max_speakers` / `max_samples_per_speaker` / `CapacityError`（§8-(d)）；超限时按 quality 与 last_seen 建议淘汰 | 一期 |
| C17 | **一致性自检** | 「爸爸」被注册了两次、`权限:监护人` 给了 8 个人 | `audit_quality()`：向量高相似跨模板、同名冲突、保留命名空间的异常用法、标签命名空间拼写建议 | 二期 |
| C18 | **与记忆/ASR 联动** | 身份的价值在于「这句话该记到谁头上」 | 输出 `SpeakerEvent{speaker_id, labels, text?, score, source}` 一行 JSONL，`memory` 模块按 `speaker_id` 归人；声纹**不**存文本，避免把身份和对话内容绑死（合规更简单） | [二期]接口，[一期]先出事件 |
| C19 | **属性标签的边界** | 自动推断性别/年龄易错且是二次隐私风险 | 只允许人工/旁路写入标签，本模块不推断、不存储推断依据 | [一期]原则 |
| C20 | **评测卫生** | 注册句同时当探针 ⇒ 指标虚高，上线才发现 | 校准/评测集与注册集按 `dedup_key` **互斥**；`evaluate()` 自动排除同 `utterance_id` 来源 | 一期 |
| C21 | **冷启动与演示模式** | 空库时全线 unknown，产品没法演 | `--backend stub` 免模型跑通全链路；空库时 `identify` 返回 `decision=UNKNOWN, ranked=()` 并在 `explain()` 里写 `gallery_empty=true` | 一期 |
| C22 | **并发与线程安全** | 注册（UI 线程）与识别（消费线程）同时写库 | store 一把 `RLock`；`matrix()` 读**快照引用**不持锁；提取器非线程安全 ⇒ 每线程一个实例或加锁（§12） | 一期 |
| C23 | **模板加密与「密钥清除式删除」** | 模板本身可被用于识别（§10.1）；只删行不删副本 = 假删除，且「删干净」在物理层面**无法自证**（EDPB 亦承认删除的技术局限，见文末合规条目） | 模板文件用本地密钥加密（`encrypt_templates`）；撤回同意时**轮换主密钥** + 删除旧密文，crypto-erasure 兜底；备份同密钥，purge 走同一路径 | 一期（有 `权限:` 标签时为强制） |

---

## 12. 延迟与算力预算

检索侧已在本机实测（numpy / 单核 / float32 归一化向量，含 `argsort` 取 top5）：

| 规模 | 打分方式 | 向量条数 | 实测 x86 | RK 量级预估 |
|------|----------|----------|----------|-------------|
| 200 人 | 质心 | 200 | 0.024 ms | ~0.2 ms |
| 2000 人 | 质心 | 2000 | **0.10 ms** | ~1 ms |
| 2000 人 × 32 样本 | 质心 | 2000 | 0.075 ms | ~1 ms |
| 2000 人 × 32 样本 | max_sample | 64 000 | **2.97 ms** | 15~30 ms |
| 10 000 人 × 32 | max_sample | 320 000 | 16.8 ms | 破预算 |
| 50 000 人 × 32 | max_sample | 1 600 000 | 86 ms | 破预算 |

内存：2000 人 × (32+1) × 192 × 4 B = **50.7 MB**（含逐样本）；只存质心 2000 × 192 × 4 = 1.5 MB。

链路其它环节：

| 环节 | 量级 | 依据 |
|------|------|------|
| `to_bundle()` | ~0.4 ms / 10s 句 | segment 侧已实测（一次解码双视图） |
| 质量门控 | < 0.1 ms | 只读 bundle 字段，不碰波形 |
| embedding 提取（CAM++ onnx，1.15 GFLOPs @ ≈2s 语音） | **没有任何官方 CPU 延迟/RTF 数据**——3D-Speaker、sherpa-onnx、WeSpeaker、SpeechBrain、NeMo 一律只给 EER / minDCF / 参数量 / FLOPs。网上流传的「1s 音频 0.10s」「RTF 是 ECAPA 的 0.4×」**无环境无出处，本文不采用** | 唯一可信做法：上板用 3D-Speaker `runtime/onnxruntime` 或 sherpa-onnx 实测 **1s / 3s / 5s 三档 × p50 / p95**，CAM++ 与 ERes2Net-base 各跑一遍。（顺带把算力门槛写在这：官方 FLOPs 里 wespeaker resnet34 = 4.55 G，是 CAM++ 1.15 G 的约 4 倍，延迟优先时它连候选都不该进。）本文档定稿时**未取得**任何一手实测（沙箱内下载 28,281,138 B 的 CAM++ onnx 时出口限速约 30 KB/s，16.8 MB 后中止） |
| 判决 + 日志 | < 0.2 ms | 纯 Python |
| 写库（注册路径） | ~ms 级 | 全量 npz 重写；2000 人可接受，更大库改增量分片或 SQLite（§14-Q3） |

**结论（有依据）**：

1. **不需要 ANN**。几百~几千模板用 numpy brute-force 微秒~毫秒级即可；faiss/HNSW 的公开指导线是万级起步，
   在 ARM 上 ANN 的额外依赖和构建成本通常超过收益；`hnswlib` 只在 >5 万条时才值得考虑。所以 v1 只做
   `centroid 矩阵 @ v_probe`，把索引优化列成「库规模 > 5 万条向量时再议」。
   另有**一组独立实测佐证同一结论**（x86_64 4 核 / numpy float32 归一化矩阵 `T@q.T`，取 7 次最小值）：
   N = 1k / 5k / 10k / 50k，D = 192 → **16.9 / 63.5 / 136.7 / 660 µs**；D = 512 → 45.7 / 176 / 348 / 2945 µs；
   10k × 192 全部向量只占 **7.68 MB**。换索引的判据应当是「**放不下内存，或打分成本超过前端推理**」，
   而不是条数本身——faiss README 的取舍框架也正是这几个维度（检索速度 / 精度 / 每向量内存 / 是否要训练）。
   `hnswlib` 侧还有两条对**本场景特别不利**的说明：`M` 强烈影响内存占用、同 id 更新更慢、**删除要 tombstone**——
   而本模块的删除与 absorb 是常态操作（§6.4 / §6.3）。
2. **真正的延迟墙是模型推理**，不是检索。RK3588 级 ARM 只有 x86 开发机约 1/4~1/8 的整数算力
   （sherpa-onnx 官方给的树莓派 5 vs x86 比例约 3.5~4×），所以 §8-(e) 的预算实际取决于 §12 第三行的实测值。
   第一期唯一的硬任务：在目标板子上量出 `extract_ms` 分布，再决定同步 / 异步（§14-Q5）。
3. `max_sample` 打分是「**精度换延迟**」：64k 向量在板子上就是十几毫秒级，比换模型更容易先撞预算，
   所以默认 `centroid`，`blend|max_sample` 只在实测「分不清」时按人/按标签局部启用。

---

## 13. 可观测性与验收

### 13.1 指标（看板必带）

```
identify:  decision 分布（known/unknown/ambiguous）、accuracy@1（校准集）、EER、FAR@FRR、
           top1−top2 margin 分布、extract_ms / search_ms p50/p95、按标签/信道的分桶准确率
enroll:    拒因分布（Verdict 计数）、每人样本数分布、quality 分布、重复注册率
库健康度:  n_speakers、DEGRADED/STALE 计数、样本分歧 top10、上次阈值标定距今、备份是否过期
风险面:    连续 unknown 峰值（可能是设备换了）、同模板分数长期下滑（老化）
```

### 13.2 测试用例（`test_voiceprint.py`，全部用 `StubExtractor`，无模型依赖）

| # | 输入 | 期望 | 覆盖 |
|---|------|------|------|
| TC1 | 注册 A(3 句)/B(3 句) → A 的一句探针 | `decision=KNOWN`，`best().speaker_id == A` | 基本 1:N |
| TC2 | 库里只有 A，探针是 C | `UNKNOWN`，`ranked` 里 A 的分数 < 阈值 | 拒识（C1） |
| TC3 | A/B 剧本相近，margin < `ambiguous_margin` | `AMBIGUOUS` 且 `best()` 不返回身份 | 模糊保护（C2） |
| TC4 | 同一 bundle 注册两次 | 第二次 `Verdict.DUPLICATE`，`n_samples` 不变 | 幂等（C3） |
| TC5 | 探针 bundle `usable_for["voiceprint"] == "short_600ms"` | 注册侧拒 `TOO_SHORT`；`allow_short_enroll=True` 时降级准入且 `quality` 打折。识别侧走 `identify(allow_short=True)`，只放宽探针下限、不改判决阈值 | §3 门控 / §8(b) |
| TC6 | 探针含 `hole_+500ms` / `zeroed_in_speech_500ms` | 拒 `SUSPECT`（补零是假样本，不许建声纹） | §3 白名单 |
| TC7 | `close_reason == "max_duration"` 的句子用于注册 | 拒 `TRUNCATED`（identify 仍可用） | §6.1 规则 1 |
| TC8 | `mean_level_db - noise_floor_db < min_snr_db` | 拒 `LOW_SNR` | C4 |
| TC9 | `sample_rate=48000` 的 bundle | `Verdict.RATE` 拒绝，不静默重采样 | §3 |
| TC10 | 库 `model_id` 与探针不同 | `ModelMismatchError`，`explain()` 说明 | §5.4 |
| TC11 | 追加样本后 `centroid == normalize(sum(samples)/n)`（逐样本精确重算） | 误差 < 1e-6 | §6.3 |
| TC12 | `absorb()` 在 `score < absorb_threshold` / 冷却内 / 与现有样本 `cos > dup_similarity` | 三种情况都不改模板，只落 `pending_absorb` | §6.3 防投毒 |
| TC13 | 打标签 `关系:家人` ×2、`权限:监护人` ×1 → `find_by_label` / `search(labels=...)` | 结果集与倒排索引一致；`search` 的 top-k 是**过滤后**的 top-k（不是全库 top-k 再筛） | §6.5 |
| TC14 | `rename_label("朋友","家人")` | 受影响人数正确、倒排索引与 `speakers.json` 一致、审计各一行 | §6.5 |
| TC15 | `delete(A)` | `who_is(A)` 抛 `SpeakerNotFoundError`；矩阵行数 −1；`search` 不命中；`export()` 无 A；`find_by_label` 无残留；`audit.jsonl` 无向量 | §6.4 级联可验证 |
| TC16 | `retire(A)` → `identify(A 的句)` → `set_state(ACTIVE)` | 停用期不命中，恢复后命中（撤销无损） | §6.4 / C6 |
| TC17 | `merge(A, B)` | 样本并集、质心 = 全体均值归一化、B 不可见、可 rollback | C7 |
| TC18 | `split(A, [a/0002])` | 新模板只含该样本，A 质心重算 | C7 |
| TC19 | `export()` → 新目录 `import_from()` | 向量/标签/状态 round-trip 全等；`on_conflict="skip"` 不覆盖 | C5 |
| TC20 | `evaluate()` 用与注册同 `utterance_id` 的探针 | 该探针被**自动排除** | C20 |
| TC21 | 并发：UI 线程 `enroll` × 消费线程 `identify` ×20 轮 | 无死锁、无半截 npz、读到的矩阵版本自洽 | C22 |
| TC22 | 两个**注册人**在探针上分数相同（stub 构造并列） | 决策不取决于 dict 迭代顺序（`sorted(ranked, key=(-score, speaker_id))`） | **决策依据已锁定为分数+speaker_id 稳定序**，见 §7.1 |
| TC23 | 门控 `tts_playing=True` 期间的 bundle 用于注册 | `Verdict.TTS_ECHO` 拒收（判定看门控状态，不看 `tag` 字符串） | C15 |
| TC24 | 连续 `unknown_prompt_after` 次 unknown（同一聚类） | 只发**一次** `RegisterPrompt` | C9 |
| TC25 | `validate()` 反例：`min_probe_ms > min_voiceprint_ms`、`dup_similarity = 1.05`、`max_speakers` 超内存预算 | 启动即 `ConfigError`（对齐 TC12@segment 的做法） | §8 不等式 |
| TC26 | `bundle_from_wav_dir()` 读 `demo3 --utt-out` 目录 | 重建出的 `suspect_reasons / zeroed_spans_ms` 与在线 `to_bundle()` 逐字段相等；`speech_spans_ms` 为空的条目被拒 `NO_SPEECH`（不许用能量兜底的 trim）；`--no-trim` 目录整体拒导入；重建样本 `synthetic=True` | §3.3 |
| TC27 | 模板带 `权限:监护人` 但 `encrypt_templates=False`；或该开关开着却设了 `utt_archive_dir` | 启动即 `ConfigError`；轮换密钥后旧密文不可解 | §8-(g) / C23 |
| TC28 | 三条时长悬殊的样本（1.5s / 4s / 8s，stub 造同一说话人、**模长随时长增大**）注册后 `recompute()` | 质心必须等于「逐条 normalize → 均值 → normalize」的结果；**与「求和后 normalize」的结果显著不同**（断言两者 cosine < 0.999，即没退化成参考实现那种长句主导）；`recompute()` 里 `abs(‖v‖−1) < 1e-3` 断言对未归一化样本确实抛错 | §6.3 求和顺序 |

### 13.3 验收（上线前）

1. 自标定集上 EER ≤ 目标值（目标值由产品定，本文不代定），且 FAR 在**权限场景**下压到可接受水平；
2. `identify` p95 延迟 ≤ `identify_budget_ms`；
3. 100% 注册动作有 `consent` 审计；删一人后 §6.4 的 5 条可验证断言全部通过；
4. `--backend stub` 下 TC1–TC28 全绿（CI 无需模型）；真后端下至少跑通「注册 2 人 + 识别 + 删除」的最小链路。

### 13.4 自测命令（落地后）

```bash
python -m unittest com.robot.voiceprint.test_voiceprint -v         # 全用例，stub 后端，无模型依赖

# 免模型跑通全链路：segment 的整句出口 → bundle_from_wav_dir() → 声纹（§3.3）
python -m com.robot.segment.demo3 --input sim --vad energy --utt-out /tmp/utt    # 先出整句 wav + manifest.jsonl
python -m com.robot.voiceprint.demo --backend stub enroll --utt-dir /tmp/utt --name 张三 --labels 关系:朋友
python -m com.robot.voiceprint.demo --backend stub identify --utt-dir /tmp/utt

# 真模型：默认后端是 onnx（CAM++ 导出，免 torch）；funasr 只用来和仓库现有 asr.py 加载路径做 A/B
python -m com.robot.voiceprint.demo enroll --utt-dir /tmp/utt --name 小明 --labels 关系:家人
python -m com.robot.voiceprint.demo --backend funasr enroll --utt-dir /tmp/utt --name 小明   # 同一批数据换后端对比
python -m com.robot.voiceprint.demo identify --utt-dir /tmp/utt
python -m com.robot.voiceprint.demo list --label 关系:家人
python -m com.robot.voiceprint.demo who  spk_7f3a
python -m com.robot.voiceprint.demo remove spk_7f3a --purge-backups
python -m com.robot.voiceprint.demo eval --gallery calib/gallery.jsonl --probe calib/probe.jsonl
```

`--utt-dir` 传的是 **demo3 `--utt-out` 的根目录**（内含 `<tag>/uttNNNNNN_*.wav` + `manifest.jsonl`），
不是裸 wav 目录——少了 manifest 就没有 `holes` / `parts[].status` 可推，`--no-trim` 出来的目录同样直接拒导入（§3.3 的两个例外）。

---

## 14. 待确认决策（需要产品 / 硬件侧回答）

**Q1. 声纹用来做什么级别的事**？只是「个性化称呼」还是「权限判定」。后者把 FAR 变成安全问题，必须叠加第二因子（§11-C14），且阈值标定要按安全口径来。

**Q2. 一人多设备/多房间**是否要求同一 `speaker_id` 跨设备可辨？影响 §11-C10 的优先级与 `设备:` 标签语义。

**Q3. 存储后端**：JSON + npz（与仓库现有 `disk_store` 同构）还是 `sqlite3`（事务更强，标准库）。v1 倾向前者，人数 > 2 万或需要并发多进程写时换。

**Q4. 是否把 segment 的整句 wav 目录挂成声纹档案**（`utt_archive_dir`）：留 = 换 embedding 模型可离线重抽迁移（§3.3）；不留 = 换模型全员重注册。开了就等于把**语音原文**的保留期与删除权同时交给声纹模块，这是合规与运维成本的直接取舍。

**Q5. 识别放同步还是异步**：取决于 §12 的 `extract_ms` 实测。同步则对话延迟 +20~60 ms；异步则先回内容、随后一句「刚才的是小明呀」。

**Q6. 家里/办公室的实际人数规模**与增长预期（决定 `max_speakers`、是否现在就上 ANN 索引，答案大概率是「不需要」）。

**Q7. 谁有权删**：`forget_by_label` 需要 `confirm_token`，token 由谁持有（App 二次确认 / 家长 PIN）。

**Q8. 校准集谁采**：§7.2 的阈值必须有真实数据标定，涉及 ≥20 人录音与知情同意——这是本模块唯一**不能靠工程单方面完成**的前置条件。

**Q9. 目标板与推理栈**：sherpa-onnx（C++，自带 `SpeakerEmbeddingManager` 参考实现、有 int8 与 arm64 APK）还是
3D-Speaker 自带的 `runtime/onnxruntime` 引擎，还是走 RKNN NPU？三者决定了 §5.1 后端表里留哪个实现。
同一批数据在 CAM++ / ERes2Net-base 上量 **1s / 3s / 5s 三档的 p50 / p95** 再定；同时必须把**权重许可的存档件**
一起结掉：页面 `License` 字段、实际**随包附带的许可正文**（CAM++ 的 `LicenseLink` 是空的，见文末存疑 12）、
以及**训练数据集协议**（WeSpeaker 跟随数据集：VoxCeleb CC-BY-4.0 可商用需署名、**CN-Celeb 明确禁止商用**，存疑 14）。
否则换了更快的模型等于换了法律风险（§5.1 坑 1）。

---

## 外部依据（本文引用的外部结论，均为 2026-11-18 抓取的二手数据）

**模型与部署**

* 3D-Speaker `iic/speech_campplus_sv_zh-cn_16k-common`：CAM++ 结构、**7.18M 参数 / 192 维**、中文 20 万说话人训练；
  官方 README EER 表（VoxCeleb1-O / CN-Celeb / 3D-Speaker）为 **0.65 / 6.78 / 7.75**，CAM++ 细分口径
  Vox1-O 0.73（LM 0.65）、MinDCF 0.091（LM 0.086）、CN-Celeb-eval 6.78 / MinDCF 0.393；
  **2026-11-19 直读模型 API 复核**：`"License": "Apache License 2.0"`、`LicenseLink` 为空、README 无「仅供研究用途」
  字样（此前本文记录的「研究用途」声明据此撤回，见存疑 12）；同一卡内表另给 **CAM++ ~200k → CN-Celeb 4.32%**
  与 CAM++ ~3k → 6.78% 两个规模 — [ModelScope 模型 API](https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh-cn_16k-common)、
  [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)、[FunASR MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)
* 官方 README 同表其它行（Vox1-O / CN-Celeb / 3D-Speaker，括号内参数量）：Res2Net 1.56/7.96/8.03（4.03M）；
  ResNet34 1.05/6.92/7.29（6.34M）；ECAPA 0.86/8.01/8.87（20.8M）；ERes2Net-base 0.84/6.69/7.21（6.61M）；
  ERes2NetV2 0.61/6.14/6.52（17.8M）；ERes2Net-large 0.52/6.17/6.34（22.46M）。
  **跨信道三项**（CAM++）：Cross-Device 7.75% / Cross-Distance 11.29% / Cross-Dialect 13.44% —
  [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)
* 官方口径里的工程常量（本文 §8 / §6.3 的依据）：训练 config `wav_len: 3.0`、`aug_prob: 0.8`（加噪 + 混响）、
  `sample_rate: 16000`；`diar.yaml` `embedding_size: 192`、fbank80 `mean_nor: True`、`mer_cos: 0.8`（聚簇合并阈值，
  非 1:N 判决阈值）；仓库自带 `runtime/onnxruntime/` C++ 引擎（opset 11、动态帧轴） —
  [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)
* sherpa-onnx 官方预导出 onnx 发布在 release tag **`speaker-recongition-models`**（36 个 asset，三条线：3D-Speaker、
  WeSpeaker、NeMo）。体积：CAM++ 28.3–29.6 MB（另有 int8）、eres2net_base 39.6 MB、eres2netv2 71.4 MB、
  eres2net_large 116.1 MB、eres2net(200k) 220.6 MB、wespeaker resnet34 26.5 MB（152/221/293 为 79/95/114 MB）、
  NeMo speakernet 23.4 MB、titanet_small 40.3 MB、titanet_large 101.4 MB。
  其 C++ `SpeakerEmbeddingManager` 是**注册 / 1:N / 删除**的参考实现：入库 normalize、全矩阵点积取 max、
  低于阈值返回 unknown、`Remove` 支持删除；但**多样本合并路径只「求和后 normalize」、未逐条归一化**（本文 §6.3 已改）；
  官方 Python / Rust 示例 `--threshold` 默认 **0.6**，示例丢弃 <0.5s 段、`min_speech_duration=0.25`；
  官方发 arm64 APK，README 支持 arm32/arm64/riscv64/树莓派/RV1126，无需 PyTorch —
  [sherpa-onnx 模型发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)、
  [speaker-embedding-manager.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/speaker-embedding-manager.cc)
* **WeSpeaker 许可红线**：官方 `docs/pretrained.md` 逐字写 "The pretrained model in WeNet follows the license of it's
  corresponding dataset. For example, the pretrained model on VoxCeleb follows *Creative Commons Attribution 4.0
  International License*"，即**模型许可跟随训练数据集**：`voxceleb_*` / `voxblink2_*` → **CC-BY-4.0（可商用、需署名）**，
  而 `cnceleb_*` 跟随 **CN-Celeb 协议**——CN-Celeb 官方页 License 段写
  "free for research institutes and individuals... **No commerical usage is permitted.**"
  ⇒ **仓库 Apache-2.0 ≠ 权重可商用；且同一个仓库里不同数据集训的权重商用结论相反**，
  选「中文声纹」时顺手取 `cnceleb_resnet34` 是本文里最容易被踩的一条线（协议原文口径见存疑 14）。
  同一页给出 params / FLOPs / EER（CAM++ 7.18M / 1.15 G；
  ECAPA c512 6.19M / 1.04 G、c1024 14.65M / 2.65 G；ResNet34-emb256 6.63M / **4.55 G**）与 AS-norm / QMF 消融
  （CN-Celeb ResNet34 EER 7.134 → 6.119、minDCF 0.408 → 0.354；vox1-O 0.867 → 0.659），并注明 `_LM` 后缀在 >3s 长语音更好；
  官方打分口径写作 `cosine (sub mean of vox2_dev)` — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)、
  [CN-Celeb 官方页](https://www.cnceleb.org/)
* 3D-Speaker **代码与配方为 Apache-2.0**，`egs/3dspeaker/sv-cam++` 是仓库里的训练/推理**配方目录**（不是 HF model id），
  2024.4 起官方给出 ONNX Runtime 部署方案与 `infer_sv.py`；sherpa-onnx 侧有 3D-Speaker 的预导出 onnx
  （`model.onnx` / `model.int8.onnx`，HF 镜像 `k2-fsa/sherpa-onnx-models`）—
  [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)、[sherpa-onnx 预训练模型页](https://k2-fsa.github.io/sherpa/onnx/speaker_recognition/pre-trained.html)
* **ERes2Net 权重页同样复核（2026-11-19）**：`iic/speech_eres2net_sv_zh-cn_16k-common` 的 `License` 也是
  `"Apache License 2.0"`、`LicenseLink` 空、无研究用途条款，`StorageSize` = **221,867,081 B**（~222 MB，即 200k 说话人规模的
  那一档）；卡内 CN-Celeb 表给 **ERes2Net ~200k → 2.79%**、CAM++ ~200k → 4.32%。
  ⚠ 本文此前写的「`damo/speech_eres2net*` 权重为 AGPL-3.0」与「HF 有 MIT 许可的 `TrainParty/eres2net_aug3`」
  **两条均不成立**：AGPL 说法与该页 `License` 字段矛盾、无出处；`https://huggingface.co/api/models?author=TrainParty`
  本轮返回**空数组**（该 org 无模型），不能当「许可干净的替代方案」引用 —
  [ModelScope ERes2Net API](https://modelscope.cn/api/v1/models/iic/speech_eres2net_sv_zh-cn_16k-common)
* 声纹后端横向对比（维度、大小、EER/minDCF、许可）：CAM++ 192 维 / 27 MB；ERes2Net 192 维 / 56 MB；
  wespeaker `RawNet3` 256 维 / 35 MB（Vox1-O 0.44%，Apache-2.0）；SpeechBrain `ECAPA-TDNN` 192 维（VoxCeleb 论文 EER 0.88%，MIT）；
  pyannote `speechbrain-lang-self-supervised-xvector` **60 维 / 2.5 MB**（离线导出 Apache-2.0，在线权重 CC-BY-4.0） —
  [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)、
  [pyannote AIHub 页](https://eu.aihub.pyannote.ai/model/speechbrain-lang-self-supervised-xvector/)、[pyannote/embedding](https://huggingface.co/pyannote/embedding)
* CAM++ 单条 1s 音频 CPU 推理约 0.10s；`infer_sv.py` 输入需 **16 kHz** — [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)
* sherpa-onnx `SpeakerEmbeddingExtractor`（C++/Python/C/Go…，官方支持 **RKNN NPU** 与 Raspberry Pi/RISC-V，Apache-2.0）；
  diarization 参考页给「2×Raspberry Pi 5 处理 3s 音频约 0.5s（CPU）」；社区实测 `eres2net_base` 在 192.6s 音频上
  15.4s（RTF≈0.08，32 线程） — [sherpa-onnx 文档](https://k2-fsa.github.io/sherpa/onnx/)、
  [预训练模型页](https://k2-fsa.github.io/sherpa/onnx/speaker_recognition/pre-trained.html)
* 短语音性能（ERes2Net-LM，Noise-ASR-V6）：5s 基线 EER 1.53% → 2.5s 2.78% → 1.25s 3.79% → **0.5s 9.27%** —
  [ModelScope sv-eres2net2907-lm 页](https://github.com/modelscope/FunASR/blob/main/docs/en/modelscope_models.md)

**判决与阈值口径**

* 3D-Speaker 文档给出 cosine 尺度 SV 阈值量级 **0.40~0.55**；ModelScope `speaker-verification` pipeline 的
  `threshold: 0.31 / 0.365` 属**内置后处理后**的分数（`score_sensevo_0.83.ckpt`），不可当裸 cosine 用；
  ERes2Net 的 CN-Celeb minDCF 阈值 0.14 同属该页自己的 score 体系 — [3D-Speaker 仓库](https://github.com/modelscope/3D-Speaker)
* 说话人日志官方 recipe 实际参数：cosine 域分数、固定凝聚阈值 **0.4**（10 维滑窗归一化空间）、
  `min_duration=0.2s`、overlap 阈值 0.8、重采样 16k — [pyannote/config.yaml](https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/pyannote/audio/pipelines/data/config.yaml)
* 模板平均顺序（本文 §6.3 采用）：**每条 utterance 先 ℓ2 归一化 → 按说话人求平均 → 再 ℓ2 归一化**；
  AS-Norm 原文还给出 cohort top-20、`μ_c/s_c=0.013/0.052`（enroll）与 `0.007/0.035`（test） —
  [AS-Norm, Mitsuda et al., Interspeech 2024](https://arxiv.org/html/2406.13708v1)
* 未做归化时 **EER 阈值不可直接当 FAR 阈值**（DFN 16k：非归化 EER 23.8% → 归化后 8.7%）；
  自适应归化在 N=300（每目标 100 trial）才可用，更小的数据要退化成仅 utterance 归化 —
  [Investigation of Adaptive Score Normalizations, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/mitsuda23_interspeech.pdf)
* 小库不建议上 PLDA：x-vector 原文用 Gaussian 对齐后**等分超平面**而非 PLDA（~1000 小时、2000+ 说话人训练） —
  [Snyder et al., x-vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)；NIST 亦建议小库 cosine 打分 —
  [NIST FRVT Part 5 Report 5z](https://pages.nist.gov/frvt/html/frvt5Z.html)
* **唯一可查到的官方示例阈值：0.6**（sherpa-onnx 说话人识别示例，Python 与 Rust 一致），其打分方式与本模块 §7.1
  同构（逐条 normalize + 全矩阵点积取 max + 低于阈值判 unknown），所以是最贴近的锚点；3D-Speaker `diar.yaml` 的
  `mer_cos: 0.8` 属**聚簇合并**阈值，不是 1:N 判决阈值。另：各家**均未发布** speaker-embedding 的 CPU 延迟/RTF，
  公开只有 FLOPs（CAM++ 7.18M/1.15 G，wespeaker resnet34 6.63M/**4.55 G**） —
  [sherpa-onnx 示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification-with-vad.py)、
  [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
* 注册/测试之间的**音素重叠**是短语音 SV 的正式评测维度（SdSV Challenge 2021 把 <3s 定义为短时长场景并系统评测它）；
  时长受限时嵌入表示不稳、对噪声与音素组成敏感 ⇒ §3.2 的「同一句切两半」防线不是洁癖 —
  [SdSV 2021 任务定义](https://arxiv.org/abs/1912.06311)、[时长受限下的表征稳定性](https://arxiv.org/abs/2606.16115)
* 嵌入**模长不是常数**，与语音时长/质量相关；quality-aware calibration（QMF）就是把时长当质量因子校准分数，
  这也是 §6.3「必须逐条归一化再平均」的直接依据 — [QMF, arXiv:2211.00815](https://arxiv.org/abs/2211.00815)
* 归化谱系与公式（Z/T/S/AS-norm，含 Sturim & Reynolds 2005、Auckenthaler 2000、Matějka 2017）汇总 —
  [arXiv:2504.04512](https://arxiv.org/abs/2504.04512)；WeSpeaker 官方消融量化了 cosine → +AS-norm → +QMF 的收益
  （CN-Celeb ResNet34 EER 7.134→6.119、minDCF 0.408→0.354；vox1-O 0.867→0.659） — 同上 pretrained.md

**存储与检索规模**

* 3000 模板 × 192 维 × float32 = **2.3 MB**（L2 归一化后点积即 cosine）；纯 numpy 单条点积在 N=1000 时约 38.4 µs —
  [SpeakerNet](https://github.com/microsoft/SpeakerNet)
* faiss 官方定位是「聚类与**大规模** k-NN 检索」，文档未给下限；社区经验线是 **50k 以上**才考虑 ANN 索引 —
  [faiss wiki](https://github.com/facebookresearch/faiss/wiki)
* **本文 §12 另给一组本机实测**（numpy 单核 float32，含 top5）：2000 人质心 0.10 ms；2000×32 逐样本 2.97 ms；
  10k×32 16.8 ms；50k×32 86 ms。与上述「小库不需要 ANN」一致。
* **第二组独立实测**（x86_64 4 核 / numpy float32，`T@q.T` 全矩阵点积，取 7 次最小值）：D=192 时
  N = 1k / 5k / 10k / 50k → **16.9 / 63.5 / 136.7 / 660 µs**；D=512 → 45.7 / 176 / 348 / 2945 µs；
  10k×192 向量共 **7.68 MB**。两组数字相互印证：亚毫秒到 10⁵ 条，瓶颈永远是前端推理。
* **换 ANN 的判据**应当是「放不下内存，或打分成本超过前端推理」，而不是条数；faiss README 自己给的取舍维度就是
  检索速度 / 精度 / 每向量内存 / 是否需要训练与添加，且 **exact search（flat）是它的基线索引**。
  `hnswlib` README 的三条对本场景尤其不利：`M` 强烈影响内存、同 id 更新更慢、**删除需要 tombstone + 内存复用** —
  而本模块的删除（§6.4）与 absorb（§6.3）是常态操作 —
  [faiss README](https://github.com/facebookresearch/faiss/blob/main/README.md)、
  [hnswlib README](https://github.com/nmslib/hnswlib/blob/master/README.md)

**工程坑**

* 44.1/48k → 16k 只保留 0–8 kHz，**上采样不恢复高频**；48k 注册 + 44.1k 使用的跨采样率对在 VoxCeleb1-O 上使
  baseline EER **2.31% → 3.22%**（+0.91），且**重采样方式的影响 > 编解码**；训练/测试混采样率时统一到最低者 —
  [Sonafx](https://sonafx.com/blogs/news/audio-sample-rates-in-ai-based-music-generation-training)、
  [SpeechBrain cookbook](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/README.md)
* 跨信道/跨设备 variability 是 speaker recognition 的经典主问题（Gobin 等） — [arXiv:1907.10968](https://arxiv.org/abs/1907.10968)
* VAD 切分位置显著影响下游：pyannote 用 **0.8** 的 overlap 判定阈值而非 0.5；F0/频谱任务建议只留语音帧、
  阈值取偏高值 — [pyannote VAD 文](https://blog.csdn.net/gitblog_01157/article/details/151504064)、
  [pyannote config](https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/pyannote/audio/pipelines/data/config.yaml)
* ECAPA **不推荐**直接用于重叠语音，官方建议改用专门重叠多说话人模型 — [speechbrain ECAPA 页](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
* 8k → 16k 属于 **train/test 带宽域失配**，电话语音专项的结论是：失配确实掉点，**首选解法是训练侧混合带宽**，
  推理侧的带宽扩展只是补救 — 对本模块的含义是「别指望重采样救回来」，所以 §3 对 `sample_rate` 不一致选**直接拒**。
  另注：SpeechBrain `classify_file` 会**自动重采样 + 自动选道**——这类便利默认值在声纹场景是隐患，
  本模块一律自己校验、不让库静默改数据 —
  [arXiv:2209.01702](https://arxiv.org/abs/2209.01702)、[arXiv:2203.16614](https://arxiv.org/abs/2203.16614)
* 跨会话（session）变异是独立于信道的一整类误差来源，两条以上长期研究的结论一致：同一人不同天的分数会整体位移 —
  [arXiv:2309.14741](https://arxiv.org/abs/2309.14741)、[arXiv:1511.06772（UAH 双源变异）](https://arxiv.org/abs/1511.06772)、
  [arXiv:1812.10260（CORAL+，指出 x-vector/i-vector + PLDA 后端的域失配需自适应）](https://arxiv.org/abs/1812.10260)
* 统计池化（attentive statistics pooling）本身带一点帧级加权，相当于**弱监督的 VAD**，
  但它不能替代前端 VAD：`speech` 视图切错边界仍会直接把分数拉下来（§11-C8 的监测项） —
  [arXiv:2405.09142](https://arxiv.org/abs/2405.09142)
* 前端一致性是阈值的隐含条件：fbank80 + `mean_nor: True` + dither 任一项变了，分数就不可比 ——
  所以 `StoreMeta` 里除 `model_id` 还要存 `feature_config`（§5.4）

**合规**

* GDPR 逐条：Art. 4(14) 生物特征定义（经特定技术处理→允许或确认唯一识别）；**Recital 51 第 3 句**（仅当经特定
  技术手段可唯一识别/认证时才落入该定义）；Art. 9(1) 原则禁止 + 9(2)(a) **明示同意**；Art. 7(3) 撤回同样容易；
  **Art. 17(1)(b) 撤回同意即须无不当延迟删除**；Art. 20(1) 可携带权**仅限**同意/合同 + 自动化处理，且须结构化、
  机器可读；**Art. 32(1)(a) 假名化与加密**；**Art. 35(3)(b) 大规模处理特殊类别须事前 DPIA** —
  [EUR-Lex 402016R0679](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng)
* PIPL 第 28 条（生物识别属敏感个人信息，须特定目的 + 充分必要 + 严格保护措施）、第 29 条（**单独同意**）、
  第 30 条（告知必要性与影响）、**第 31 条（不满 14 周岁须监护人同意并制定专门处理规则）**；
  第 47 条（删除权）原文本次**未核** — [PIPL 全文镜像](https://www.wicinternet.org/pdf/Personal%20Information%20Protection%20Law%20of%20the%20People%20Republic%20of%20China.pdf)
* 同类产品怎么落地（可直接抄的三条）：**Google Voice Match** — 停用即「从设备删除模型 + 从账号删除录音」，
  设备侧生效最长 24 h，不活跃模型 12 个月过期，并明确写「在某些地区语音模型可能被视为生物特征数据」；
  **Amazon Alexa** — 云端不留存录音则关联 voice ID **一并删除且功能直接失效**，可按条/日期/设备/voice ID 删，
  3 或 18 个月自动删除可配；**Apple Siri** — 尽量设备端处理、用随机标识符而非账户、不保留音频除非用户主动共享 —
  [Google 账号 Voice Match](https://support.google.com/accounts/answer/16467537)、
  [Google Nest 语音匹配](https://support.google.com/googlenest/answer/15559809)、
  [Alexa 隐私 FAQ](https://www.amazon.com/gp/help/customer/display.html?nodeId=GVP69FUJ48X9DK8V)、
  [Apple Siri 隐私承诺 2025-01](https://www.apple.com/newsroom/2025/01/our-longstanding-privacy-commitment-with-siri/)
* 「仅语音不用于识别时可能不属生物特征数据」「删除在技术上难以自证」「**撤回必须真的触发删除**」为 EDPB 立场 —
  [GDPR Recall](https://gdprrecall.com/articles/biometric-data-gdpr/)
* 中国 PIPL 第 28 条（生物识别属敏感个人信息）、第 29 条（**单独同意**）、第 30 条（告知必要性与影响） —
  [PIPL 全文](https://www.wicinternet.org/pdf/Personal%20Information%20Protection%20Law%20of%20the%20People%20Republic%20of%20China.pdf)
* NIST SP 800-63-3 §5.2.3.1：生物特征模板不使用受保密保护密钥时**不得作为认证器**，存储/传输须加密或哈希 —
  [NIST 63-3](https://pages.nist.gov/800-63-3/sp800-63-3.html)
* 模板不可逆推语音但**可直接用于识别**，「240× 压缩」不是安全措施 — [pm-research 模板保护文](https://pm-research.com/content/ijsigsp12/3/10.13169/isjnsgsp12.3.0205.full/)

**存疑 / 不可靠数据点**（引用前须复核，本文一律按「量级」而非承诺值使用）

1. CN 侧 EER 口径此前打架：CN-CVS 6.64%（仅见博客）vs 6.74%（模型卡）vs CN-Celeb 6.78%（官方 README 表）。
   **已按 3D-Speaker README 表统一为 CN-Celeb 6.78%（CAM++，~3k 说话人训练）**；本文早先写的「CN-Celeb 6.61%」是把
   ERes2Net-base 的**参数量 6.61M** 误当成 EER，已订正。**不要用博客数字做效果承诺。**
   ⚠ 补充订正（2026-11-19 直读模型卡）：ModelScope 卡内表同一行还给了 **CAM++ ~200k → 4.32%**，
   即「4.32%」不是博客编的、而是**另一个训练规模**（200k 说话人）的官方数字；两数并存时**必须连训练说话人数一起引用**，
   否则 6.78% 与 4.32% 看起来互相矛盾。本文正文一律用 ~3k 的 6.78%，因为它对应公开可复现的配方。
   同一口径下 ERes2Net 也有 **~200k → 2.79%**，但那一档权重 `StorageSize` = 221.9 MB（板上部署用的是 39.6–71.4 MB 的
   base/V2），**规模与数字不可交叉引用**。
2. **已确认是「没有」而不是「没找到」**：3D-Speaker / sherpa-onnx / WeSpeaker / SpeechBrain / NeMo 全部**未发布**
   speaker-embedding 的 CPU RTF 或单条延迟基准，公开可比的只有 FLOPs。⇒ §12 的「上板实测」是硬任务，不是可选项。
3. 中文集 + 产业界 cosine 阈值的「0.40~0.55」是**文档级口径**，非论文表格实测对照（社区报告同族阈值约 0.32~0.46）。
   官方侧只有两个可查数字：**示例默认 0.6** 与 **`mer_cos: 0.8`（用途不同）**。本文的 0.45 占位值明确标注为未标定。
4. 网传 HF 镜像 `3d-speaker/sv-cam++`（27 MB onnx、Vox1-O 0.79%、CN-CVS 6.64%）**复核失败**：
   `huggingface.co/3d-speaker` 在 2026-11-18 返回 404，`sv-cam++` 实际是 GitHub 仓库里的配方目录。
   本文只把「27 MB」当量级参考；Apache-2.0 目前只能确认到**代码与配方**层面，权重能不能商用必须回源逐条确认。
5. sherpa-onnx 侧 2025/2026 新增模型（`titanet-small`、`tita_net_resnet293`）无 EER/大小/许可记录。
6. 「嵌入是否可逆」学界与厂商口径不一致（不可反推 vs 可检出共享特征）；本文按「保守：不可泄露」处理。
7. 没有任何后端**官方给出**最短语音时长硬性要求。最接近官方口径的只有：3D-Speaker 训练 `wav_len: 3.0`（训练裁剪窗口，
   不是推理下限）、sherpa-onnx 示例丢弃 <0.5s 段且 `min_speech_duration=0.25`。
   所以「注册 3~5s × 3 句、检索 ≥ 2~3s、<1s 不用」是**工程外推**；本文的 `min_enroll_ms=3000` / `min_probe_ms=600`
   同样不是厂商承诺。中文 8–16k 短语音的「经验阈值区间」**未找到任何有方法论的可靠来源**。
8. 「短语音嵌入不稳」有**两条同源/同向**支撑：3D-Speaker 中文短语音实验表（5s 1.53% → 0.5s 9.27%）与
   SdSV Challenge 2021 对 <3s 场景的正式评测（音素重叠为主因，arXiv:1912.06311）、时长受限下表征不稳定的分析
   （arXiv:2606.16115）。但常见引用的「VoxCeleb 在 <2s 显著退化」**未找到原始表格**，本文不作依据。
9. 未取得的数据：NeMo `titanet_small/large`、`speakernet` 的 EER / 参数量 / 输出维度（NeMo results 页已改版）；
   `pyannote/embedding` 的输出维度与参数量（卡片未标，且 gated 需 HF token）；SpeechBrain ECAPA 的参数量
   （卡片未标，同类为 6.19M / 14.65M / 20.8M）；FunASR `.bin` 权重体积（约 90 MB，未验证）；ModelScope 侧
   HF 镜像页 401/404。这些后端在 §5.1 表里只能留空，不许用外推填。
10. 「getdsemv3」这一模型名在 sherpa-onnx 官方 asset 列表与公开检索中**不存在**，判断为误记；最接近的是
    3D-Speaker `eres2netv2`，或 CosyVoice-ONNX 里的 `campplus.onnx` + `speech_tokenizer_v3.onnx` 组合。
    凡是挂这个名字的数字一律不采纳。
11. 明确不采用的博客数字：「CAM++ 5s 音频 CPU <100 ms」、
    「RTF 为 ECAPA 的 0.4×」（两者均无环境无出处）、阈值区间 0.7–0.85 / 0.31→0.42→0.55–0.68、
    「归化队列需 N ≥ 20000」（均无方法论来源）。「pyannote 需 ~2 GB torch、模型 ~100 MB」出自教学仓库而非官方。
    Azure 说话人识别的「≤300 秒注册 / 每请求 ≤50 人 / 16bit-16kHz」来自搜索缓存片段，且该服务已于
    **2025-09-30 下线**，属历史口径，只可当产品形态参考、不可当技术约束引用。
12. **本文作废的一条自述结论**：早先版本称「`damo/speech_campplus_sv_zh-cn_16k-common` 权重页写『所有权重文件均仅限
    研究用途，不可用于商业用途』」。2026-11-19 直读 ModelScope 模型 API 复核：`damo` 与 `iic` 解析到同一 model_id，
    `"License": "Apache License 2.0"`，README 全文无该句，`LicenseLink` 为空。
    ⇒ 该说法**撤回**。但它只是换成另一个不确定：**没有可核对的许可正文**，且与 FunASR 仓库 `MODEL_LICENSE`
    的口径并存，**出厂以随包文本为准**。凡是据此前说法做的「必须换权重」决策，现在应重新评估（见 §14-Q9 与
    `docs/multi_speaker_pipeline_design.md` §9）。
    同批撤回两条：「`damo/speech_eres2net*` 权重为 AGPL-3.0」（该页 `License` 实为 Apache License 2.0，AGPL 说法无出处）、
    「HF 存在 MIT 的 `TrainParty/eres2net_aug3` 可作许可干净的替代」（`api/models?author=TrainParty` 返回空数组）。
13. sherpa-onnx 各 onnx 权重**发布页自身不写 license 段落**（仅 release 资产 + 上游 ModelScope/HF 页面），
    2025–2026 新增的 `titanet-small` / `tita_net_resnet293` 亦无许可记录；`ClearerVoice-Studio` 预训练权重许可未声明。
     ⇒ 「派生打包层」的许可判断只能靠回源上游，不能引用 sherpa-onnx 的 Apache-2.0 代替。
14. **数据集层已确认的一条硬红线**：CN-Celeb 官方页 License 段写
    "All the resources contained in the dataset are free for research institutes and individuals... **No commerical usage is permitted.**"
    （本次抓到的是 **CN-Celeb-AV** 协议页；**CN-Celeb 1/2 的协议原文本轮未单独取到**，同课题组发布、口径推定一致，
    但引用前必须补取原文）。WeSpeaker 的 `cnceleb_*` 权重跟随该协议 ⇒ **不可商用**，与 `voxceleb_*`（CC-BY-4.0）不同。
15. 中文数据/权重链条仍未闭环的三处：`iic/SenseVoiceSmall` 与 Paraformer 的训练集写作未公开的 industrial 数据；
    `wenetspeech.com` 许可页本轮抓取失败（502/DNS），只有 GitHub README 自述 CC-BY-4.0；
    WavLM 上游 `microsoft/wavlm-large` 在 HF **没有 license 标签**（凡是 DiariZEN 一类派生都要连上游一起查）。
