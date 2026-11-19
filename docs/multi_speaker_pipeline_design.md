# 多人说话识别链路（multispeaker）— 技术设计文档

> 状态：**设计稿 v1，未落库，未实现**。对应需求文档 `docs/multi_speaker_pipeline_requirements.md`。
> 本文只回答「怎么实现」：类、接口、参数、判定规则、许可。边界语义以需求文档 §4 为准。
> 本文**取代** `docs/speaker_pipeline_design.md` §2 拓扑与 §5 编排器；那份文档的 §4 接口签名继续作为组件目录，本文标注复用/作废。

---

## 1. 拓扑与不变式

```
bundle ──► Gate ──► Counter ──► Router ──┬─ ZERO/UNCERTAIN(低) ─► 只 ASR
（segment）（计数）（路由）              ├─ SINGLE ─► Anchors ─► [已知 ─ 记活动 | 未知 ─ Provisional] ─► ASR
                                         ├─ MULTI/sequential ─► SplitRequester ─► 回 segment 重切
                                         └─ MULTI/simultaneous ─► Extractor ─► LeakageGate ─► ActivityTracker
                                                                     │                          │
                                                                  residual ─► (可选)新建临时声纹 │
                                                                                               ▼
                                                                    每路 IASRBackend ─► SpeakerText
```

四条**硬不变式**（CI 可断言）：

| # | 不变式 | 违反后果 |
|---|--------|----------|
| I1 | `SINGLE` 路径不出现任何 `Extractor`/`residual` 调用 | 计数早退失效，延迟翻倍 |
| I2 | `Counter` 只读 bundle，不改 bundle，不调 ASR | 计数被伪影污染 |
| I3 | 残差进聚类前必须过 `LeakageGate` | 「假第三人」（需求 E13） |
| I4 | 任何 `Provisional` 生成点都要过 `may_create_provisional()` | TTS/外放/suspect 句污染身份库 |

---

## 2. 类清单

包名 `com/robot/multispeaker/`（对齐 `com/robot/segment`、`com/robot/voiceprint` 的一层命名）。

| 类 | 职责 | 依赖 | 可 stub | 说明 |
|----|------|------|---------|------|
| `MultiSpeakerPipeline` | 门面：编排 + 路由 + 汇总 + `explain()` | 下面全部 | ✅ | 产品只碰这个 |
| `IBundleGate`(`AsrEchoGate`) | 预门控：静音/TTS 回声/采样率 | bundle, segment gate | ✅ | 复用 voiceprint C15 判定 |
| `ISpeakerCounter`(`WindowEmbeddingCounter`) | **主判据**：窗嵌入 + 锚点矩阵 → `CountEstimate` | `IWindowExtractor`, `IAnchorProvider` | ✅ | 唯一计数出口 |
| `IWindowExtractor`(`BackendWindowExtractor`) | 把 bundle 切成窗并出向量 | 声纹 embedding 后端 | ✅ | 与 voiceprint §5.1 同一后端实例 |
| `IEvidenceProbes`(`SpectralEvidenceProbe` / `SpatialEvidenceProbe`) | 佐证证据（双 F0 / DOA 多峰） | numpy 或麦阵 | ✅ | 单通道时 `SpatialEvidenceProbe` 返回 `unavailable` |
| `IAnchorProvider`(`VoiceprintAnchors`) | 提供已知 + 临时质心矩阵 | voiceprint 门面 | ✅ | 只读，不注册 |
| `IProvisionalRegistry`(`InMemoryProvisionalRegistry`) | 临时声纹生命周期（§8） | — | ✅ | 默认不落盘 |
| `ISeparator`(`TseSeparator`) | 目标提取（已知/临时声纹条件） | TSE 权重 | ✅ | **默认关闭**（§5 `separation_enabled`，§10 未实测） |
| `ILeakageGate`(`CosLeakageGate`) | 残差窗 vs 已提取人 打分判泄漏 | `IWindowExtractor` | ✅ | I3 的执行者 |
| `IActivityTracker`(`MatrixActivityTracker`) | 软分配 × VAD → 每人活动线 | score 矩阵 | ✅ | 「谁在说」的唯一来源 |
| `IASRBackend`(`FunAsrBackend`/`StubAsr`) | 单路识别，可注入、无状态机 | ASR 权重 | ✅ | 从 `asr.py` 抽出（§13-P2） |
| `ISplitRequester`(`SegmentSplitRequester`) | 向 segment 发重切请求 | segment | ✅ | 需求 E02 |
| `PipelineMetrics` | 计数分布/早退命中率/泄漏拦截数/丢弃率 | — | ✅ | N6 |

**作废/复用**：旧文档 §4.1 `IAudioPreprocessor`（本链路不做二次分帧，直接用 bundle 帧视图）、§4.3 `IBeamformer`+§4.2 `IDOAEstimator`（降级为可选证据探针）、§4.5 `ISpeakerMatcher`（并入 `IAnchorProvider`）、§4.8 `IUnknownSpeakerClusterer`（仅残差分支复用）、§4.10/4.11 `ConfidenceFilter`/`ResultAggregator`（并入门面）、§5 `SpeakerPipeline`（本文件 §3 取代）。§4.6 `ITargetSpeakerExtractor` 的**签名作废**（`target` 是 `speaker_id` 不够，需向量），§4.7 `IResidualCalculator` 保留但强制后接 `ILeakageGate`。

---

## 3. 接口（ABC）

```python
# com/robot/multispeaker/interfaces/counter.py
class ISpeakerCounter(ABC):
    @abstractmethod
    def count(self, bundle: "UtteranceBundle",
              anchors: Optional[np.ndarray],      # (A, D) 归一化；None = 空库
              evidence: Evidence) -> CountEstimate:
        """I2：不得修改 bundle，不得调 ASR/Extractor。anchors 为空矩阵时仍能出 MULTI。"""

# interfaces/extractor.py
class IWindowExtractor(ABC):
    @abstractmethod
    def windows(self, bundle, window_ms: int, hop_ms: int) -> WindowEmbeddings: ...
    @property
    @abstractmethod
    def model_id(self) -> str: ...               # 与 voiceprint §5.4 同源，混版本直接 raise
    @property
    @abstractmethod
    def dim(self) -> int: ...

# interfaces/separator.py
class ISeparator(ABC):
    @abstractmethod
    def extract(self, bundle, target_vector: np.ndarray) -> SeparatedTrack: ...
    @abstractmethod
    def extract_multi(self, bundle, targets: Sequence[np.ndarray]) -> List[SeparatedTrack]:
        """K 路必须并行还是串行由实现决定；返回顺序与 targets 一致。未启用时 raise DisabledError。"""

# interfaces/gates.py
class ILeakageGate(ABC):
    @abstractmethod
    def judge(self, residual: WindowEmbeddings,
              extracted: Sequence[np.ndarray]) -> LeakageVerdict:
        """I3。返回 kept_windows / leaked_windows，绝不返回「新说话人」。"""

class IBundleGate(ABC):
    @abstractmethod
    def admit(self, bundle) -> GateDecision:      # pass / drop(reason)
    def may_create_provisional(self, bundle, estimate) -> Tuple[bool, str]: ...  # I4

# interfaces/activity.py
class IActivityTracker(ABC):
    @abstractmethod
    def track(self, score_matrix: np.ndarray,     # (W, A+K)
              frame_flags: FrameFlags,
              speakers: Sequence[str]) -> List[Activity]: ...

# interfaces/asr.py
class IASRBackend(ABC):
    @abstractmethod
    def recognize(self, samples: np.ndarray, sample_rate: int) -> AsrPiece: ...
        # 幂等、无跨调用状态；不许自带 VAD 状态机（asr.py 现状）
    @property
    @abstractmethod
    def license_verified(self) -> bool: ...        # §9 硬闸门

# interfaces/split.py
class ISplitRequester(ABC):
    @abstractmethod
    def request(self, req: SplitRequest) -> List["UtteranceBundle"]:
        """segment 侧按 boundaries_ms 重切；返回条数 < 2 视为失败，退回整句单人处理。"""
```

门面：

```python
class MultiSpeakerPipeline:
    def __init__(self, cfg: MultiSpeakerConfig, *, counter, anchors, windows, gate,
                 separator=None, leakage=None, activity=None, asr=None,
                 provisional=None, splitter=None, clock=time.monotonic): ...
    def process(self, bundle) -> MultiSpeakerVerdict: ...   # 唯一同步入口，不抛业务异常
    def explain(self, verdict) -> str: ...                  # F10：凭什么这么多人 / 凭什么说他 / 这路为什么丢
    def stats(self) -> dict: ...
```

构造注入 + 每个协作者都有 `Stub*`，保证 N7（无模型可跑通全路径）。

---

## 4. DTO

```python
# com/robot/multispeaker/models.py（设计稿）
class CountLevel(Enum):    ZERO = "zero"; SINGLE = "single"; MULTI = "multi"; UNCERTAIN = "uncertain"
class CountMode(Enum):     NONE = "none"; SEQUENTIAL = "sequential"; SIMULTANEOUS = "simultaneous"
class IdentityKind(Enum):  KNOWN = "known"; PROVISIONAL = "provisional"; NONE = "none"
class Confidence(Enum):    HIGH = "high"; MEDIUM = "medium"; LOW = "low"
class Evidence(IntFlag):   # F1 的证据位图，explain 直接读它
    SPATIAL = 1; SPECTRAL = 2; EMBEDDING = 4; ANCHOR = 8
    UNAVAILABLE_SPATIAL = 16; DEGRADED = 32          # E11 / E18

@dataclass(frozen=True)
class WindowEmbeddings:
    start_ms: Tuple[int, ...]        # 每窗起点，与 bundle 时间轴同基
    vectors: np.ndarray              # (W, D) float32，逐行 L2 归一化（voiceprint §6.3 的求和坑同此）
    usable: Tuple[bool, ...]         # 该窗是否够长够干净；不可用窗不参与簇结构判定
    model_id: str
    def n_usable(self) -> int: ...

@dataclass(frozen=True)
class CountEstimate:
    level: CountLevel
    mode: CountMode
    confidence: Confidence
    evidence: Evidence
    overlap_ratio: float             # 帧级同时发声占比，仅 MULTI/SIMULTANEOUS 有意义
    n_upper: int                     # 需求 §1.2：只给上界，不给「就是 4 个人」
    score_matrix: Optional[np.ndarray]   # (W, A) 窗×锚点 cosine；单人不认识时可能为 None
    cluster_gap: Optional[float]     # 主判据的簇分离度，标定用
    degraded: str = ""               # "counter_unavailable" 等（E18）
    elapsed_ms: float
    def allows_provisional(self) -> bool: ...   # 与 GateDecision 求与，见 I4
    def explain(self) -> str: ...

@dataclass(frozen=True)
class Activity:
    speaker_id: str                  # 已知 id 或 tmp_* 
    identity_kind: IdentityKind
    spans_ms: Tuple[Tuple[int, int], ...]
    coverage: float                  # 发声帧 / 该人应有的语音帧
    mean_score: float
    def duration_ms(self) -> int: ...

@dataclass(frozen=True)
class SeparatedTrack:
    speaker_id: str
    identity_kind: IdentityKind
    samples: np.ndarray              # float32 ±1.0，与 bundle 同采样率
    quality: float                   # 提取质量分（SINR 代理或能量保真）
    activity: Optional[Activity]
    discarded_by: str = ""           # "leakage" / "low_quality" / "coverage"

@dataclass(frozen=True)
class LeakageVerdict:
    leaked: Tuple[int, ...]          # 判为泄漏的残差窗下标
    kept: Tuple[int, ...]
    max_target_cos: float
    verdict: str                     # "clean" / "leakage" / "new_voice"

@dataclass(frozen=True)
class SpeakerText:
    text: str
    speaker_id: Optional[str]
    identity_kind: IdentityKind
    score: float
    confidence: float
    activity: Optional[Activity]
    overlap: bool                    # 需求 E17：重叠帧文本必须显式标注
    source_quality: str              # "direct" / "separated" / "residual"
    utterance_id: int
    def as_dict(self) -> dict: ...   # 不含音频

@dataclass(frozen=True)
class SplitRequest:
    utterance_id: int
    boundaries_ms: Tuple[int, ...]   # 簇切换点，严格递增且落在整句内
    reason: str                      # "sequential_multi"
    dedup_key: str                   # 沿用 voiceprint §3.2，保证 E19 幂等

@dataclass(frozen=True)
class MultiSpeakerVerdict:
    estimate: CountEstimate
    texts: Tuple[SpeakerText, ...]
    provisional_ids: Tuple[str, ...]
    discarded: Tuple[str, ...]       # 每个丢弃一项 "<speaker_id>:<reason>"
    events: Tuple[Tuple[str, dict], ...]     # ("suggest_enroll", {...}) / ("provisional_merged", {...})
    elapsed_ms: float
    suspect: bool
    def speaker_for(self, idx: int) -> Optional[str]: ...   # 「他刚才说」的指代入口（G3）
    def as_dict(self) -> dict: ...
```

`MultiSpeakerVerdict` 不落音频，身份可失效时 `speaker_id` 保留字符串 + `identity_kind=NONE`（需求 E15）。

---

## 5. 参数表与硬不等式

```python
# com/robot/multispeaker/config.py（设计稿）
@dataclass
class MultiSpeakerConfig:
    # 输入对齐
    sample_rate: int = 16000
    channels: int = 1
    view: str = "full"                 # 计数/分离用 full，ASR 用 speech（segment §5.6.1）

    # 计数主判据
    window_ms: int = 1500
    hop_ms: int = 750
    min_windows_for_multi: int = 2     # 少于此不许判 MULTI（E01 的硬支撑）
    intra_max_cos_single: float = 0.55 # 窗间相似低于此 ⇒ 嵌入层证据成立（未标定）
    min_cluster_silhouette: float = 0.10
    min_snr_for_multi_db: float = 8.0  # 低于此最多判 UNCERTAIN（E05）
    evidence_votes_for_multi: int = 2  # 几层证据才许 MULTI；默认 2 ⇒ 偏向「1」

    # 锚定
    anchor_top_k: int = 8
    anchor_threshold: float = 0.45     # 与 voiceprint identify_threshold 同源（未标定占位）
    anchor_ambiguous_margin: float = 0.05
    overlap_min_ratio: float = 0.25    # 同时判定所需重叠帧占比

    # 分离
    separation_enabled: bool = False   # N2：未上板实测前默认关
    max_extract_targets: int = 3
    min_track_quality: float = 0.55
    min_track_coverage: float = 0.30
    leakage_cos: float = 0.60          # 残差窗与该人 cos ≥ 此 ⇒ 判泄漏（E13）
    residual_cluster_enabled: bool = False   # 残差新建临时声纹，风险最高，默认关

    # 临时声纹
    provisional_enabled: bool = True
    provisional_ttl_s: int = 900
    provisional_max_per_session: int = 8
    provisional_min_ms: int = 1500     # 短于此不建（E01）
    provisional_persist: bool = False  # Q1：默认不落盘

    # 活动与 ASR
    activity_tol_ms: int = 120
    asr_min_coverage: float = 0.35     # 不达标不送 ASR（E16）
    asr_parallel: int = 1

    # 预算
    single_budget_ms: float = 30.0
    multi_budget_ms: float = 1200.0
    mem_budget_mb: int = 8             # 只算本模块新增缓冲，模型权重另计（见 (g)）
    max_utterance_ms: int = 8000

    # 许可闸门（§9）
    license_gate: bool = True

    def validate(self) -> None: ...
```

`validate()` 里的不等式全部是**断言级**，不满足直接构造失败：

```
(a) 输入对齐   sample_rate == SegmentConfig.sample_rate；channels ∈ {1, 2}
(b) 窗结构     hop_ms ≤ window_ms / 2
               window_ms ≤ max_utterance_ms
               判 MULTI 的前提：floor((dur_ms − window_ms)/hop_ms)+1 ≥ min_windows_for_multi
               ⇒ 这条就是 E01 的代码化：窗数不够时任何实现都到不了 MULTI
(c) 证据单调   evidence_votes_for_multi ≥ 1；且「层数 < 阈值」时禁止输出 MULTI（只允许 UNCERTAIN）
(d) 泄漏更严   leakage_cos > anchor_threshold  且  leakage_cos < 1.0
               否则 I3 失去意义（要么全判泄漏、要么永远判不出泄漏）
(e) 目标上限   1 ≤ max_extract_targets ≤ 3 且 ≤ provisional_max_per_session
(f) 质量下限   min_track_quality > 0 且 asr_min_coverage ≤ min_track_coverage 时报警（两个门控不能互相矛盾）
(g) 内存       W_max × dim × 4B + (K+2) × max_utterance_ms × sr/1000 × 4B
               + provisional_max_per_session × dim × 4B + score_matrix ≤ mem_budget_mb
               实算（dim=192, K=3, 8s/16k float32）：音频缓冲 (3+2)×0.49 ≈ 2.5 MB 是主项，
               窗向量与临时表合计 < 20 KB ⇒ 本模块新增 ≈ 3 MB。
               ⚠ 只覆盖本模块新增缓冲；模型权重常驻（onnx ≈ 28 MB 量级，见声纹文档 §5.1）由
               voiceprint 侧预算承担，两处不能混算也不能都不算。
(h) TTL        provisional_ttl_s ≥ 60 且 provisional_persist=False ⇒ provisional_enabled 才可 True
(i) 预算       single_budget_ms < multi_budget_ms；separation_enabled=False 时 multi_budget_ms 不参与断言
(j) 许可闸门   license_gate=True 时，每个 enabled=True 的模型组件 license_verified 必须为 True，
               否则 raise LicenseError（§9 出厂硬规则）
```

---

## 6. 计数判定矩阵（主判据 + 佐证）

`WindowEmbeddingCounter.count()` 的实际逻辑，六步，全部可 explain：

```
1. 出窗       windows = IWindowExtractor.windows(bundle, window_ms, hop_ms)
2. 簇结构     两两 cos → 若 min(usable 间 cos) < intra_max_cos_single 且 silhouette ≥ min_cluster_silhouette
              ⇒ EMBEDDING 证据；否则无
3. 锚点打分   score_matrix[w, a] = windows[w] · anchors[a]（anchors 含已知 + 本会话临时质心）
              · 同一行两个锚点同时 ≥ anchor_threshold 且差 < margin 且持续 ≥ 2 窗 ⇒ ANCHOR（同时）
              · 行间 top1 切换 ≥ 1 次 ⇒ ANCHOR（时序）
4. 佐证       频谱：双 F0 同时成立的帧占比 ≥ 阈值 ⇒ SPECTRAL
              空间：DOA 多峰（峰间隔 ≥ min_doa_sep）⇒ SPATIAL；单通道 ⇒ UNAVAILABLE_SPATIAL
5. 票数       证据层数 ≥ evidence_votes_for_multi ⇒ MULTI，否则 SINGLE；层数 == 阈值−1 或质量不足 ⇒ UNCERTAIN
6. 定 mode    ANCHOR/嵌入 指向同一行高分 → SIMULTANEOUS；指向跨行切换 → SEQUENTIAL；
              冲突时取 SIMULTANEOUS（保守：宁可多一次分离，不可漏分离）
```

不对称代价体现在两处：第 5 步的票数阈值默认 2（要两层证据才敢拆人），第 6 步 mode 冲突取保守值。

`n_upper` = 簇数估计上界 = `min(max_extract_targets + 1, 4)`，>3 人只报上界（需求 E08）。

边界到规则的映射（每行在 §12 有对应 TC）：

| 需求 | 落点规则 | 实现位置 | TC |
|------|----------|----------|-----|
| E01 | (b) 窗数不足 ⇒ 禁 MULTI + 禁建临时声纹 | `validate` + `count` 第 2 步前置 | TC1 |
| E02 | mode=SEQUENTIAL ⇒ 只发 `SplitRequest`，不进 `ISeparator` | 路由表 R2 | TC2 |
| E03 | mode=SIMULTANEOUS ⇒ 逐 target 提取 + `overlap_ratio` | 路由表 R3 | TC3 |
| E04 | 计数只看帧级矩阵，不看静态 top-k 排序 | 第 3 步「同一行」约束 | TC4 |
| E05 | SNR/时长不足时上限 `UNCERTAIN` | 第 5 步质量分支 | TC5 |
| E06 | TTS 回声在预门控拦，计数之前 | `IBundleGate.admit` | TC6 |
| E07 | 外放：禁止残差建临时声纹 + `环境:` 标签 | `may_create_provisional` + `residual_cluster_enabled=False` | TC7 |
| E08 | >3 人 ⇒ `n_upper` 且只提 K 路 | (e) + 第 6 步排序 | TC8 |
| E09 | 无语音 ⇒ `ZERO`，不调模型 | `admit` | TC9 |
| E10 | suspect bundle ⇒ 计数可跑、禁注册/absorb | `may_create_provisional` | TC10 |
| E11 | 空间证据缺席 ⇒ `UNAVAILABLE_SPATIAL` + 阈值收紧 | 第 4 步 + 第 5 步票数不变、质量收紧 | TC11 |
| E12 | 同锚点分数位移不算多人证据；窗内一致性与跨锚点区分度分开算 | 第 3 步只用「同一行多锚点」/「行间 top1 切换」 | TC12 |
| E13 | 残差先过 `LeakageGate`（I3） | `judge` + (d) | TC13 |
| E14 | `merge_provisional` 不回写正式模板、可回滚 | §8 状态机 | TC14 |
| E15 | TTL 回收 ⇒ 保留字符串 + `identity_kind=NONE` | §8 reclaim | TC15 |
| E16 | 覆盖率/质量不达标不送 ASR；文本带 `source_quality` | (f) + `asr_min_coverage` | TC16 |
| E17 | 重叠帧文本 `overlap=True` 且置信降级 | `SpeakerText` | TC17 |
| E18 | 后端缺失 ⇒ `degraded=counter_unavailable` + 单人路径 | `count` 兜底 try/except | TC18 |
| E19 | `dedup_key` 幂等：重放不新增临时向量 | §8 create 前查 | TC19 |
| E20 | `unknown_prompt_after` 触发「建议注册」事件，不自动注册 | §8 事件 + 需求 §1.2 | TC20 |

路由表（门面的唯一分支逻辑）：

```
R1  admit=drop                      → MultiSpeakerVerdict(texts=(), suspect=False)
R2  SINGLE + KNOWN                  → Activity(known) → ASR(speech)          [I1：无分离]
R3  SINGLE + UNKNOWN + may_create   → Provisional.create → ASR
R4  SINGLE + UNKNOWN + ¬may_create  → ASR，identity_kind=NONE
R5  MULTI + SEQUENTIAL              → SplitRequester → 逐条按 R2/R3/R4 处理
R6  MULTI + SIMULTANEOUS + 分离启用  → ISeparator.extract_multi → LeakageGate → Activity → 每路 ASR
R7  MULTI + SIMULTANEOUS + 分离关闭  → 整句 ASR 一次，所有帧标 overlap=True，identity 取矩阵 top1 或 NONE  [Q2]
R8  UNCERTAIN                       → ASR + count_confidence，不建临时声纹
R9  ZERO                            → 空结果
R10 counter 异常                    → 视同 R8 且 degraded=counter_unavailable     [E18]
```

---

## 7. 分离、泄漏门控与活动线

```
targets（按 质量×活动时长 排序，截到 max_extract_targets）
  ├─ 已知声纹质心（voiceprint anchors，ACTIVE 优先，PROVISIONAL 次之）
  └─ 本会话临时声纹质心
        │
        ▼  ISeparator.extract_multi(bundle.full, targets)
   tracks[K]  ──► 质量门控（min_track_quality / min_track_coverage）──► 丢弃记入 discarded
        │
        ▼  residual = full − Σ tracks        （旧文档 §4.7 保留）
   ILeakageGate.judge(residual_windows, tracks 的向量)
        ├─ "leakage"   → 丢窗，禁止建身份（E13）
        ├─ "clean"     → 无
        └─ "new_voice" → 仅当 residual_cluster_enabled 且 may_create_provisional 才建 tmp
        │
        ▼  MatrixActivityTracker.track(score_matrix × tracks 软分配, frame_flags, speakers)
   Activity[per speaker] = spans_ms / coverage / mean_score
```

三点设计取舍：

* **TSE 而非盲分离**。targets 由声纹条件化，失败时退化为「只标 overlap」（R7），不会出现「拆出 4 路但没人知道是谁」。盲 BSS（SepFormer 一类）不进主链路：算力大、需要 N 先验、且拆出来的还是匿名流（需求 §1.2）。
* **门控放在聚类之前**。旧文档 §2 是 `Residual → Clusterer` 直连，这是「假第三人」的结构性来源；本文把它改成 `Residual → LeakageGate → Clusterer`，且 `Clusterer` 默认关闭。
* **活动线与分离同源**。`Activity` 由 score 矩阵 × 软分配导出，不再跑第二次 diarization——「谁在说」和「拆给谁」必须是一套证据，否则两处口径迟早漂（需求 F6/G4）。

质量门控用的 `quality` 是代理量（能量保真 + 残差占比 + 与目标声纹的 cos），**不是**SINR 真值；标定前不进需求 §7 的验收指标。

---

## 8. 临时声纹生命周期

```
             create(tmp_id, vector, src)                  touch()
   (无) ───────────────────────────────► PROVISIONAL ─────────────┐
                    ▲                       │      │              │
                    │ may_create=True       │      │ merge(人工确认)│ TTL 到期
                    │ (I4 五道门)            │      ▼              ▼
                    │                  reject│  MERGED ──► 指代改写 │ RECLAIMED
                    │                        ▼  (可回滚)           │ identity_kind=NONE
                    └── 会话结束 / reclaim ◄──┘                    ▼
                                                            (向量丢弃，不留痕)
```

`tmp_id = "tmp_<session_uuid>_<seq>"`，标签命名空间保留 `临时:`（voiceprint §6.5）。

`may_create_provisional()` 的五道门（I4，任一不过即拒）：

1. `estimate.level == SINGLE` 且 `confidence != LOW`；
2. bundle 无 `suspect_reasons`（E10）、非 `max_duration`/`aborted` 截断；
3. 语音时长 ≥ `provisional_min_ms`（E01）；
4. 门控非 TTS 回声（E06）；残差分支还需 `residual_cluster_enabled`（E07/E13）；
5. `dedup_key` 未出现过（E19）且未超 `provisional_max_per_session`。

与 voiceprint 的衔接（需求 §8 要求上游的改动，全部为**新增**，不改既有语义）：

| 改动 | 位置 | 说明 |
|------|------|------|
| `TemplateState.PROVISIONAL` | voiceprint §4 枚举 | 不参与 `export()` / `find_by_label()` |
| `identify(include_states=(ACTIVE, PROVISIONAL))` | voiceprint §5.3 | 默认仍是 `(ACTIVE,)`，本链路显式 opt-in |
| `merge_provisional(tmp_id, speaker_id)` | voiceprint §5.3 | 只改指代与丢弃临时向量，**不回写正式模板**（分离来源不可信）；带逆操作以支持回滚（E14） |
| `SourceRef.via_pipeline=True` | voiceprint §4（字段已有） | 本文开始真正使用：匹配可用、注册/absorb 禁止 |
| `promote(tmp_id, speaker_id, consent=True)` | voiceprint §6.1 | 临时 → 正式的唯一通道，必须两步式 + 同意 |

`provisional_persist=False`（默认）时向量只在内存；`audit.jsonl` 记 create/merge/reclaim 三类动作，**不含向量**（N5）。

---

## 9. 商用许可

这一章是出厂条件，不是附录。三层链条 = **代码许可 / 权重许可 / 训练数据集许可**；任何一层写「仅研究用途」或查不到书面出处，该组件即不得随产品分发。

### 9.1 硬规则

```
L1 每个模型组件必须实现 license_verified()，返回 True 的前提是三层都有官方 URL + 抓取日期 + 书面确认。
L2 config.license_gate=True（默认）时，validate() 对每个 enabled 组件断言 L1，不满足 raise LicenseError。
   ⇒ 「许可没核实」在代码层面等价于「这个开关不许开」，不靠人记。
L3 未闭环的组件必须以默认关闭形态出厂，且在 §14 风险清单里点名。
L4 权重不进仓库：走 provision/烧录，出厂清单（SBOM）逐个记 model_id + 来源 + 许可文本摘要。
L5 再分发（含 APK/OTA）比纯内部使用更严：数据集链条（如需署名/非商业/共享同意的）必须逐条落实到产品条款。
L6 兜底：IWindowExtractor / ISeparator / IASRBackend 全部可注入，license_gate 让「许可」与「能力」解耦——
   许可没闭环就换权重或关功能，不改上层逻辑。这条是本文敢把 §9.2 一半以上标 ⚠ 的前提。
```

### 9.2 逐环节链条（⚠ 标记的条目为 2026-11-19 本轮直读官方页面/API 核实）

| 环节 | 具体权重 | 代码 | 权重 | 训练数据集 | 判定 |
|------|---------|------|------|-----------|------|
| 声纹 embedding | sherpa-onnx 预导出 onnx（`speaker-recongition-models`） | Apache-2.0 | 发布页**不写** license 段落，一律回源上游 | 同上 | ⚠ 不改变上游判定 |
| 声纹 embedding | 3D-Speaker `CAM++` / `ERes2Net`（ModelScope `iic`/`damo` 同一 model_id） | Apache-2.0 | ⚠ 页面 `License` 字段写 **Apache License 2.0**、`LicenseLink` **为空**；但 FunASR 仓库另有 `MODEL_LICENSE` v1.1（ bespoke：2.1 自由使用/复制/修改/分享、**无 NC 条款**；2.2 须注明出处并**保留模型名**；3 句「**仅供参考和学习使用**」为免责表述；5 违约即自动终止）| ⚠ 卡面只写「约 200k 说话人的大型中文数据集」，**无数据集许可** | ⚠ **两处口径冲突**（字段 Apache-2.0 vs 随附 bespoke 协议）⇒ 出厂以随包文本为准；数据层不可举证 |
| 声纹 embedding | WeSpeaker `voxceleb_*` / `voxblink2_*` | Apache-2.0 | ⚠ 官方 `pretrained.md` 逐字：**跟随数据集许可** → **CC-BY-4.0** | VoxCeleb CC-BY-4.0（署名） | ✅ **唯一三层都有书面出处的一条**（需把署名义务写进产品条款） |
| 声纹 embedding | WeSpeaker `cnceleb_*` | Apache-2.0 | 同上「跟随数据集许可」 | ⚠ CN-Celeb 官方协议原文：**"No commerical usage is permitted."** | ⛔ **不可商用**（「选中文声纹就顺手拿 CN-Celeb 权重」是最容易踩的线） |
| VAD | Silero VAD | ⚠ LICENSE 正文本轮直读：**MIT**（"Copyright (c) 2020-present Silero Team"，含 sell 权利，无 NC/研究限制） | 同文件覆盖「software and associated documentation files」，**权重单列声明未找到** | 「6000+ 语种大语料」，清单未公开 | ✅ 首选；~2 MB 级、有 ONNX、streaming |
| VAD / 分段 | funasr `iic/*` FSMN-VAD | MIT（FunASR 仓库） | FunASR `MODEL_LICENSE` v1.1（同上，非 Apache） | "industrial" 数据，未公开 | ⚠ 沿用现链路的，按风险自留处理 |
| 计数 / 重叠（可选件，本文不启用） | `pyannote/segmentation-3.0` | pyannote-audio MIT | ⚠ API `cardData.license = "mit"`，gating 文案自证 "Though this model uses MIT license and will always remain open-source" | 卡片列 AISHELL/AliMeeting/AMI/AVA-AVD/DIHARD/ECLAT/Ego4D/MSDWild/REPERE/VoxConverse，**逐项许可未声明** | ⚠ 权重层干净；**断点在数据层 + gating 合同**（`gated:"auto"` 需填公司/网址，官方保留邮件营销）。⚠ `usedStorage` 仅 **5.9 MB**，板端算得起 |
| 计数 / 重叠 | DiariZEN(`diarizen-wavlm-*`)、`nvidia/diar_sortformer_4spk-v1` | MIT / NeMo | ⚠ **CC-BY-NC-4.0**（且上游 `microsoft/wavlm-large` 在 HF **无 license 标签**） | LDC/AMI/ICSI/AISHELL-4 等 | ⛔ 不可商用；且非板端算力路线 |
| 分离 / TSE | `speechbrain/sepformer-libri2mix`·`libri3mix` | Apache-2.0 | ⚠ `cardData.license = "apache-2.0"`，`datasets:["Libri2Mix"]`，`language:"en"`（ckpt `usedStorage` ≈ **319 MB**） | Libri2Mix → LibriSpeech，⚠ OpenSLR SLR12 官方页 **License: CC BY 4.0** | ✅ 许可链条最完整的分离权重 / ⚠ **英文训练**，中文重叠零验证 |
| 分离 / TSE | `sepformer-wsj02mix`·`wsj03mix`、`wham`/`whamr` 系、`iic/speech_mossformer_separation_temporal_8k` | Apache/MIT | 字段常标 apache-2.0（MossFormer 页 `License` = Apache License 2.0、`LicenseLink` 空、`StorageSize` 169,158,745 B） | ⚠ WHAM! 官方页 License 段 = **CC BY-NC 4.0**，且原文写 "users will need access (and license) to the WSJ0 dataset"；⚠ MossFormer 卡内**中文自述**：「混合语音是基于 WSJ0 数据集生成的，**由于 WSJ0 的 License 问题无法在这里分享**」 | ⛔ **权重页标 Apache 也别用**：限制来自数据层，官方自己已经承认 |
| 分离 / TSE | WeSep（wenet-e2e） | Apache-2.0 | **官方权重未发布**（roadmap "coming soon"，ONNX 导出亦未支持） | 配方用 LibriMix / VoxCeleb2Mix（CC-BY-4.0）/ MC-LibriMix | ⚠ 商用路线成立但**必须自训**，不是拿来即用 |
| 分离 / TSE | VoiceFilter / SpEx / SpEx+ / SpeakerBeam 官方实现 | ⚠ 仓库 404 / 无 LICENSE | 未找到 | 多为 WSJ0-2mix | ⛔ 只能当论文参考 |
| ASR 中文 | `emiltiar/wenet-uv-docs-conformer-zh-78M-1221`（WeNet U2++ Conformer CTC） | WeNet Apache-2.0 | HF `cardData.license` = **Apache-2.0** | WenetSpeech（GitHub README 自述 CC-BY-4.0，**许可页本轮 502 未复核**）+ People's Speech BSD + VoxCreed CC0 | ⚠ **最有希望全绿的一条**，差 WenetSpeech 那一页；CTC 非自回归、CPU 友好 |
| ASR 中文 | 现用 `iic/SenseVoiceSmall`（`asr.py` 默认） | MIT（FunASR） | ⚠ HF `cardData.license="other"`、`license_name="model-license"` → 即 FunASR `MODEL_LICENSE` v1.1；ModelScope 侧 `StorageSize` 940,019,376 B | ">40 万小时" 工业数据，未公开 | ⚠ 无 NC 条款，但 bespoke + 数据层不可举证 |
| ASR 中文 | Whisper `large-v3` | MIT | ⚠ 官方 README 直述「**code and model weights are released under the MIT License**」 | 1M h 弱标注 + 4M h 伪标注（YTC 来源） | ⚠ 权重层最干净，**1.55B 参数板端不现实** ⇒ 本项目不适用 |
| ASR/分离 | NVIDIA NeMo 各模型 | NeMo Apache-2.0 | ⚠ **逐模型不同**：Sortformer `CC-BY-NC-4.0`、Nemotron-3 `openmdw-1.1`、multitalker-parakeet `nvidia-open-model-license`；NGC 侧另受 NGC ToS | 含 LDC / YTC 混合 | ⚠ **不能一句「NeMo 开源可商用」带过**，NGC 与 HF 两条分发路径条款不同 |

### 9.3 这一轮把两个旧结论推翻了（必须写进文档，否则下次照旧决策）

* ⛔→✅ 「`damo/*` 声纹权重仅研究用途」**不成立**：ModelScope `damo`/`iic` 解析到同一 model_id，`License` 字段是
  `Apache License 2.0`，README 全文无该句。**极可能的来源**是 FunASR `MODEL_LICENSE` 第 3 条那句
  「[FunASR 软件]**仅作为参考和学习使用**」——那是**免责条款**（不承担损失），不是许可范围限制；
  但正因为这句话存在，保守读法仍有争议空间 ⇒ 交给法务定性，工程侧只负责把**随包文本**存档。
* ⛔→✅ 同批撤回：「`damo/speech_eres2net*` 权重为 AGPL-3.0」（该页 `License` 实为 Apache License 2.0，AGPL 无出处）、
  「HF 上存在 MIT 的 `TrainParty/eres2net_aug3` 可作干净替代」（`api/models?author=TrainParty` 返回**空数组**）。
* ⇒ 真正的阻塞点换成了另外三条：**(a)** 中文重叠语音**没有现成可商用的 TSE 权重**（英文 LibriMix 系能商用但中文零验证，
  WSJ0/WHAM 系明确不可商用，WeSep 要自训）；**(b)** CN-Celeb 系声纹权重**禁止商用**；
  **(c)** FunASR 系（SenseVoice / Paraformer / fsmn-vad / CAM++）的**训练数据层无法举证**，
  且页面 License 字段与随附 bespoke 协议口径不一致。

### 9.4 三套可落地的组合（按取舍列，不推荐「最好的」，只推荐「你知道自己在承担什么」）

| 组合 | 内容 | 代价 | 适用 |
|------|------|------|------|
| **A 举证完全** | Silero VAD(MIT) + 计数用**自有权重**：WeSpeaker `voxceleb_*`(CC-BY-4.0) 或自训 embedding + ASR 走 WeNet 中文 CTC（Apache 权重 + CC-BY/BSD/CC0 数据，待 WenetSpeech 页闭环）+ **不做同时多人分离**（多人走 R7：整句 ASR + `overlap=True`） | 多人路径功能弱化（需求 Q2 的「跑不起/不启用」分支成为默认）；无中文 TSE | 法务一票否决制、出海产品 |
| **B 工程省事（当前建议默认）** | 现状 FunASR 栈（fsmn-vad + SenseVoice + CAM++）+ 声纹侧改用 `voxceleb_*` 或保留 CAM++ 并由法务确认 bespoke 读法；重叠分离**只开英文可用的 `sepformer-libri*` 做实验、不进出厂配置** | 数据层举证缺口留自留风险；`License` 字段与 bespoke 协议必须择一固化并随包 | 先出内部版/试点，同时并行补举证 |
| **C 能力上限** | WeSep(Apache) + LibriMix/VoxCeleb2Mix(CC-BY-4.0)/AliMeeting(Apache) **自训**中文 cue-conditioned TSE + 自采/自合成中文重叠语料 | 需要训练投入与标注集；DS-TALC 一类「现成中文重叠数据」官方标 **research-only，不许用来训商用模型** | 决定长期做多人时的正解 |

**明确不要做**：把 `sepformer-wsj02mix` / `wham` / `whamr` / DiariZEN / Sortformer / CN-Celeb 系声纹任何一个塞进出厂镜像；
把 gated 的 pyannote 权重当「Apache 随便用」；把 NeMo 整体当可商用。

### 9.5 出厂 checklist（发布流水线逐条打勾）

1. SBOM 列出每个 `model_id`、文件 SHA256、来源 URL、抓取日期。
2. 三层链条各有官方许可原文，且写明「商用 / 再分发 / 是否需署名 / 是否非商业 / 是否共享同许可」。
3. 数据集链条含第三方语料授权（WSJ0 / LibriSpeech / VoxCeleb / **CN-Celeb** / 3D-Speaker / WenetSpeech）的确认记录。
4. `validate()` 在 CI 里跑一次 `license_gate=True` 全开配置，任何缺项即构建失败。
5. 未闭环组件的功能开关在出厂配置里为 `False`，并出现在 §14 风险清单与客户可见文档中。
6. 隐私侧联动：临时声纹默认不落盘（`provisional_persist=False`）+ 未同意不产生正式模板 + `export()`/`find_by_label()` 查不到 `tmp_*`（voiceprint §10、C23）。
7. **字段与正文择一固化**：页面 `License` 字段与实际随附的许可正文不一致时（CAM++ / SenseVoice / Paraformer / fsmn-vad
   均属此类：字段写 Apache-2.0，仓库里另挂 FunASR `MODEL_LICENSE` v1.1），出厂镜像里**只留一份正文**，
   并在 SBOM 里注明「以随包正文为准」；两份并存 = 未闭环。
8. 署名义务落地：选了 WeSpeaker `voxceleb_*`（CC-BY-4.0）就必须能在产品的「开源许可/第三方声明」页找到 VoxCeleb 署名条目。
9. gated 模型（pyannote 系）如启用，须留 HF 条款同意记录与申请主体，**且不得把权重转存进自有分发**（合同层义务，不只是技术问题）。

> 注：§9.2 的 ⚠/⛔ 判定均来自 2026-11-19 直读官方页面或 API 的原文；本轮**未取到原文**的一律列在文末存疑，
> 并按 L2/L3 视为不可启用。

---

## 10. 延迟与算力预算

| 分支 | 构成 | 预算 | 依据 |
|------|------|------|------|
| 预门控 | bundle 元信息判定 | <1 ms | 无模型 |
| 计数（1 人） | 1 窗 embedding + 矩阵点积（A≤8） | ≤ `single_budget_ms` = 30 ms（**待实测**） | 点积项可忽略：窗向量 W×D + A×D，微秒级 |
| 计数（≥2 窗） | W=⌈dur/hop⌉ 次 embedding | dur=5s → W≈5 | 每窗 ≈1.5s 音频，**单次前向耗时是全部预算的不确定性来源** |
| 分离 | K 路 TSE | ≤ `multi_budget_ms` | TSE 是链路最贵一步，未实测前 `separation_enabled=False`（N2） |
| 残差 + 门控 | residual 出窗 + K×W 点积 | 十毫秒级 | numpy 可算 |
| ASR | 每路一遍 | 与路数线性相关 | `asr_parallel=1` 起步 |

诚实口径：**本文不写任何未实测的 CPU 毫秒数**。embedding 各家均未发布 CPU 延迟基准，声纹文档 §5.1 同样记为「未找到可靠数据」；上板实测（RK 级，CAM++ 同规模 onnx 的 1s/3s/5s 前向）是 Q2 与 `single_budget_ms` 的前置条件。

内存按 §5-(g) 断言：主项是 K 路分离缓冲（8s/16k float32 ≈ 0.49 MB/路，K=3 加混合与残差共 ≈ 2.5 MB）；模型权重本身是另一个量级，由声纹侧预算承担。

---

## 11. 落库结构

```
com/robot/multispeaker/
├── __init__.py           # re-export：Pipeline/Config/DTO + Stub 工厂
├── models.py             # §4 DTO（纯标准库 + Enum/IntFlag）
├── config.py             # §5 MultiSpeakerConfig.validate()
├── pipeline.py           # MultiSpeakerPipeline：路由表 R1–R10、explain、stats
├── counter.py            # WindowEmbeddingCounter + 窗/簇/锚点打分
├── evidence.py           # SpectralEvidenceProbe / SpatialEvidenceProbe（惰性 numpy）
├── provisional.py        # InMemoryProvisionalRegistry：create/touch/merge/reclaim（RLock）
├── separator.py          # ISeparator 实现 + DisabledError
├── gates.py              # AsrEchoGate / CosLeakageGate / may_create_provisional
├── activity.py           # MatrixActivityTracker
├── asr_backend.py        # IASRBackend：从 com/robot/models/asr/asr.py 抽出（前置条件 P2）
├── split.py              # SegmentSplitRequester（调 segment 重切入口）
├── backends/             # onnx / sherpa / stub 注入点，许可开关集中在这里
└── stubs.py              # 全部协作者的 stub 实现（N7：无模型跑通全路径）

var/multispeaker/         # 默认只有指标与审计，不存音频
├── metrics.json          # 计数分布/判 MULTI 比例/早退命中率/泄漏拦截数
├── audit.jsonl           # 一行一动作：create/merge/reclaim/discard，不含向量与音频
└── provisional.dump      # 仅当 provisional_persist=True（Q1）时存在，加密 + TTL
```

写盘沿用仓库既有做法：逐项 JSON + 索引、临时文件 + `os.replace`、`RLock`。

---

## 12. 测试用例（全部用 stub，可在 CI 无模型跑）

| # | 场景 | 断言 |
|---|------|------|
| TC1 | 1.2s 单窗 bundle | `level != MULTI`；不建 tmp（(b)/E01） |
| TC2 | 窗向量两簇 + 行间 top1 切换 | `MULTI/SEQUENTIAL`；`ISeparator` **零调用**（I1）；发出 1 次 `SplitRequest` |
| TC3 | 同一行双锚点高分 | `MULTI/SIMULTANEOUS`、`overlap_ratio>0` |
| TC4 | 库有 A/B，只有 A 发声 | `SINGLE`，B 不计入（E04） |
| TC5 | SNR=3 dB 双簇向量 | `UNCERTAIN` 而非 `MULTI`（E05） |
| TC6 | `gate.tts_playing=True` | `admit=drop`；计数不执行（E06） |
| TC7 | 残差含外放人声 | `residual_cluster_enabled=False` ⇒ 无新 tmp（E07） |
| TC8 | 5 簇输入 | `n_upper=4`，提取路数 ≤3（E08） |
| TC9 | 纯噪声 | `ZERO`，模型调用计数为 0（E09） |
| TC10 | `suspect_reasons=["hole_120ms"]` | 计数出结果、`may_create=False`（E10） |
| TC11 | `channels=1` | `evidence` 含 `UNAVAILABLE_SPATIAL`，仍可判 SINGLE/MULTI（E11） |
| TC12 | 同一锚点分数逐窗下移 | 不产生 ANCHOR 证据（E12） |
| TC13 | 残差窗与已提取人 cos=0.75 | `LeakageVerdict="leakage"`，`Clusterer` 零调用（I3/E13） |
| TC14 | `merge_provisional` + 回滚 | 指代改写；正式模板向量不变；回滚后 tmp 可再用（E14） |
| TC15 | TTL 到期后读历史 | `speaker_id="tmp_..."` 保留、`identity_kind=NONE`（E15） |
| TC16 | coverage=0.2 的路 | 不送 ASR，进 `discarded`（E16） |
| TC17 | R7 关闭分离 | 文本 `overlap=True` 且置信降级（E17） |
| TC18 | `count()` 内部抛异常 | `degraded=counter_unavailable`，仍出 `SpeakerText`（E18） |
| TC19 | 同 bundle 投喂两次 | tmp 表长度不变（E19） |
| TC20 | 连续 5 次 unknown 单人 | 出 `suggest_enroll` 事件，**无正式模板**（E20） |
| TC21 | `leakage_cos ≤ anchor_threshold` | `validate()` raise（(d)） |
| TC22 | `min_windows_for_multi=2` 且 dur<window | 任何 stub 输入都无法产出 MULTI（(b) 可达性） |
| TC23 | `license_verified=False` + `separation_enabled=True` | `LicenseError`（L2） |
| TC24 | `provisional_persist=True` | `validate()` raise 或 `provisional_enabled` 强制 False（(h)） |
| TC25 | `explain()` 全分支 | 每个决策点至少一句可读文本（F10） |
| TC26 | 全 stub 端到端 | 无 funasr/sherpa/faiss 环境下 R1–R10 全走通（N7） |

自测：`python -m com.robot.multispeaker.selftest --fixture-dir tests/multispeaker/fixtures`（fixtures 为 `.npz`：窗向量 + 帧标志，不含真实音频）。

---

## 13. 分期

| 期 | 内容 | 交付判据 |
|----|------|----------|
| P1 | `models/config/gates/counter(嵌入+锚点)/provisional/activity/pipeline` + 全 stub + TC1–TC26 | R1–R5、R8–R10 可用；I1–I4 有断言；单人路径真实延迟上板实测 |
| P2 | `IASRBackend` 从 `asr.py` 抽出 + 许可核实；segment 侧新增切句请求入口 | R5（时序多人）真链路可跑 |
| P3 | `ISeparator` + `ILeakageGate` 上板实测 + 标定；频谱/空间证据探针 | R6 可开；`single/multi_budget_ms` 有实测数 |
| P4 | 残差聚类与 `promote` 产品化、阈值标定集与 DET、指标看板 | 需求 §7 验收 1/5 条有数 |

**P1 就要定权重落点，不等 P4**：§9.3 把「damo 声纹权重不可商用」这条伪阻塞撤掉之后，真阻塞换成了
「FunASR 系的 License 字段与随附 bespoke 正文口径不一致」+「中文重叠语音没有现成可商用的 TSE 权重」。
前者决定阈值标定算不算数（换后端 = 迁移，声纹文档 §5.4），后者决定 §9.4 里选 A 还是 C。
P1 结束前必须有一次产品/法务共同签字的选型定案，否则标定与分离两条线都在做白工。

## 14. 本文的不足与风险

* 计数的**绝对精度无法承诺**：`intra_max_cos_single`、`min_cluster_silhouette`、`overlap_min_ratio` 全是未标定占位，必须用自建多说话人中文集（含时序/同时两类标注）标定。
* 频谱证据（双 F0）在中文声调语言上判别人声调与谐波易混，可能长期只能当弱证据；届时票数阈值需重算，否则第 5 步等价于只有两层。
* **Q2 未决**：板端跑不起 TSE，则多人路径退化为 R7，`谁在说` 只剩整句 top1 —— 功能面比本文承诺的 G4 弱很多，需产品知情。
* 许可：**中文重叠语音不存在「现成可商用 + 板端跑得动」的 TSE 权重**（§9.4 三条路各有代价），这是能力面而非合规面的硬约束；
  另外 FunASR 系权重的 `License` 字段与随附 bespoke 正文不一致、CN-Celeb 系声纹权重禁止商用（§9.2 ⛔ 行），
  任一条没闭环都会被 §5-(j) 的 `license_gate` 卡在出厂前。
* `ASR` 侧现状是脚本（模块级加载 + 自带 VAD 状态机 + `print`），P2 未完成前本文的「每路独立 ASR」只是接口上的成立。
* E07 外放/电视只做了「禁止建身份」，没有识别能力；产品层面仍需播放状态等先验。
* `n_upper` 与残差聚类相互独立性有限，>3 人时可能既漏又错，只保证不硬拆。

---

## 15. 待确认（沿续需求文档编号）

* **Q1** 临时声纹是否落盘 → 本文默认 `provisional_persist=False`，(h) 已把它变成硬约束。
* **Q2** TSE 上板可行性 → §13 P3 + §10，实测前 `separation_enabled=False`。
* **Q3** 外放/电视（E07）要不要产品级规避 → 本文只做「禁止对残差建身份 + `环境:` 标签供人工修正」，不做外放检测；若要结合播放状态/音量先验，属 `IBundleGate` 的扩展点。
* **Q4** `max_extract_targets` → 本文取 3，(e) 断言上限。
* **Q5** 是否引入专用计数模型 → 本文**不引入**，先用「窗嵌入 + 锚点矩阵 + 频谱/空间佐证」；若上板后 MULTI 召回不足再评估，且必须过 §9 的许可关卡。
* **Q6** 时序多人由谁裁句 → 本文按「segment 侧裁」实现（`ISplitRequester`），本模块不复制一套切句逻辑。

---

## 外部依据

标「本轮直读」= 2026-11-19 由本文作者直接抓官方页面/API 并引原文，不是转述博客。

* 本轮直读 ModelScope 模型 API：`damo`/`iic` 两路径同为 `iic/speech_campplus_sv_zh-cn_16k-common`，
  `"License": "Apache License 2.0"`、`LicenseLink: ""`、README 无研究用途限制句；卡内 CN-Celeb 表给
  CAM++ ~3k = 6.78% / ~200k = 4.32% — https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh-cn_16k-common
  （ERes2Net 同口径：`License` 亦 Apache-2.0、`StorageSize` 221,867,081 B —
  https://modelscope.cn/api/v1/models/iic/speech_eres2net_sv_zh-cn_16k-common）
* 本轮直读 FunASR `MODEL_LICENSE`（v1.1，中英双语）：2.1 允许自由使用/复制/修改/分享、**无 NC 条款**；
  2.2 须注明出处作者并保留模型名；「仅作为参考和学习使用」出现在**第 3 条责任条款**里；5 违约自动终止 —
  https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
* 本轮直读 WeSpeaker `docs/pretrained.md` 原文："The pretrained model in WeNet follows the license of it's
  corresponding dataset… VoxCeleb follows Creative Commons Attribution 4.0 International License" —
  https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
* 本轮直读 CN-Celeb 官方页 License 段："free for research institutes and individuals… **No commerical usage is permitted.**"
  （抓到的是 CN-Celeb-AV 协议页） — https://www.cnceleb.org/
* 本轮直读 `pyannote/segmentation-3.0` API：`cardData.license = "mit"`、`gated: "auto"`、gating 文案自证 MIT、
  `usedStorage` 5,905,440 B — https://huggingface.co/api/models/pyannote/segmentation-3.0
* 本轮直读 `speechbrain/sepformer-libri2mix` API：`license: apache-2.0`、`datasets:["Libri2Mix"]`、`language: en`、
  `usedStorage` 318,841,933 B — https://huggingface.co/api/models/speechbrain/sepformer-libri2mix
* 本轮直读 OpenSLR SLR12（LibriSpeech）：**License: CC BY 4.0** — https://www.openslr.org/12/
* 本轮直读 WHAM!/WHAMR! 官方页：License 段 = **CC BY-NC 4.0**，且原文 "users will need access (and license) to the
  WSJ0 dataset" — http://wham.whisper.ai/
* 本轮直读 `iic/speech_mossformer_separation_temporal_8k`：`License` Apache-2.0、`LicenseLink` 空、
  `StorageSize` 169,158,745 B、`Datasets` 指向 `modelscope/Libri2Mix_8k`，卡内自述 WSJ0 生成数据
  「由于WSJ0的License问题无法在这里分享」 —
  https://modelscope.cn/api/v1/models/iic/speech_mossformer_separation_temporal_8k
* 本轮直读 Silero VAD `LICENSE` 正文 = **MIT**（"Copyright (c) 2020-present Silero Team"，含 sell 权利） —
  https://raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE
* 本轮直读 `microsoft/wavlm-large` API：`cardData` **无 license 字段**、tags 内无 `license:*`（`usedStorage` ≈ 3.79 GB）—
  https://huggingface.co/api/models/microsoft/wavlm-large
* 未标「本轮直读」的条目（WeSep roadmap、NeMo 各模型 `license_name`、WeNet 中文 CTC 卡片、SenseVoice `StorageSize`、
  Moonshine Mandarin Community License、DiariZEN / Sortformer 的 CC-BY-NC）来自本轮许可核查的抓取记录（2026-11-18）—
  https://github.com/wenet-e2e/wesep 、https://huggingface.co/api/models/nvidia/diar_sortformer_4spk-v1 、
  https://huggingface.co/emiltiar/wenet-uv-docs-conformer-zh-78M-1221 、
  https://huggingface.co/FunAudioLLM/SenseVoiceSmall 、https://github.com/moonshine-ai/moonshine/blob/main/LICENSE
* 3D-Speaker（代码 Apache-2.0；训练 `wav_len: 3.0`、`aug_prob: 0.8`） https://github.com/modelscope/3D-Speaker
* sherpa-onnx 代码 Apache-2.0 与 onnx 权重发布 https://github.com/k2-fsa/sherpa-onnx
* VoxCeleb 数据集 CC-BY-4.0 https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ ；WenetSpeech README 自述 CC-BY-4.0
  https://github.com/pengzhendong/WenetSpeech
* 声纹侧阈值 / 指标 / 许可细节见 `docs/voiceprint_pipeline_design.md` §5.1、§7.2、§11-C15、§10

## 存疑（不作为依据）

1. **已撤回的三条自述结论**（留在这里防止再被引用）：① 「`damo/*` 声纹权重仅限研究用途」；
   ② 「`damo/speech_eres2net*` 权重为 AGPL-3.0」；③ 「HF 存在 MIT 的 `TrainParty/eres2net_aug3` 可作干净替代」
   （`api/models?author=TrainParty` 返回空数组）。缘由见 §9.3 与声纹文档文末存疑 12。
2. **LibriMix 的 GitHub 仓库本轮无法确认**：`api.github.com/repos/epic-spadre/librimix` 与
   `api.github.com/repos/Jorisos/LibriMix_script` **均 404** ⇒ 本文只引 OpenSLR 的 LibriSpeech（CC BY 4.0）
   与 ModelScope 的 `Libri2Mix_8k` 条目，不给 LibriMix 仓库 URL。
3. **CN-Celeb 1/2 的协议原文本轮未单独取到**（抓到的是 CN-Celeb-AV 页）；同课题组发布、口径推定一致，
   但引用「不可商用」前必须补取 CN-Celeb 原始协议。
4. **WenetSpeech 许可页本轮抓取失败**（wenetspeech.com 502 / DNS），只有 GitHub README 自述 CC-BY-4.0
   ⇒ §9.4 组合 A/C 的中文 ASR 链条**差这一页没闭环**，暂不计入「三层齐备」。
5. **Silero「MIT 从哪个版本起适用」无官方逐版本声明**（版本表只给时间与格式）；ONNX / safetensors 权重是否落在
   LICENSE 的 "software and associated documentation files" 之内属解释而非明文 ⇒ 合规文档里只能表述为
   「当前 LICENSE 为 MIT」，或向 Silero 取书面确认。
6. **ModelScope 的 `License` 字段不等于许可正文**：CAM++ / ERes2Net / MossFormer 三页的 `LicenseLink` 全为空；
   sherpa-onnx 各导出页自身也不写 license 段落；`ClearerVoice-Studio` 权重许可未声明
   ⇒ 「派生打包层」一律回源上游，不许拿代码许可当权重许可。
7. 「板端 CPU 能否实时 TSE / 分离」无任何可靠公开基准，本文不给数；§10 的 30 ms / 1200 ms 是**设计目标**，非实测。
8. 双 F0 证据在中文上的可用性只有文献层面的怀疑，没有本仓库实测；`language: "en"` 的分离权重用于中文重叠属**零验证**。
9. 窗长 1.5s / hop 0.75s 取自声纹模型 3s 工作尺度与 diarization 常规做法的折中，非本场景标定值。
