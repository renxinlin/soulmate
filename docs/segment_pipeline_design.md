# 实时语音活动分段（VAD Segmentation）— 设计文档

> 模块位置（建议）：`com/robot/segment/`
> 上游：音频采集（pyaudio / ALSA / pulseaudio 桥接）
> 下游：`com/robot/models/asr`（SenseVoice + sherpa-onnx）、`com/robot/speaker/pipeline`（见 `docs/speaker_pipeline_design.md`）
> 场景：实时对话机器人，双状态（说话 / 空闲）

---

## 0. 需求原始描述（保留原文口径）

| # | 原始规则 |
|---|----------|
| R0 | 语音持续采集到 6 秒环形队列 |
| R1 | VAD 游标位于写指针之前 0.5 秒 |
| R2 | 滑动窗口为 2 秒 |
| R3 | 启动时为空闲 |
| R4 | 静音大于 0.3 秒 → 空闲 |
| R5 | 说话超过 5 秒 → 截断，状态机置空闲 |
| R6 | 没有长声音，或长声音小于 0.3 → 保持空闲 |
| R7 | 说话，且这段话的起点大于游标且大于 0.3 → 状态机转说话；之后需将一整段话送入后续处理 |

本文档在 §2 给出术语修正，在 §5 给出与 R3–R7 一一对应的状态机，在 §7 给出 R1–R7 的不足清单与补全方案。**§7 是本文档最重要的部分**。

---

## 1. 设计目标与非目标

**目标**
1. 在 6 秒环形缓冲约束下，把连续音频流切成「语义上完整、可整段送 ASR」的语音段（utterance segment）。
2. 判决延迟可控且有上界：起口延迟 ≤ 1.0 s，闭合延迟 ≤ 0.9 s（见 §6 延迟预算）。
3. 判决可复现：所有阈值/时间量集中在一个配置对象，所有状态转移可被单测逐帧驱动。
4. 不丢帧、不重复派发、不因缓冲回绕覆盖未消费数据。

**非目标（本期不做，但接口预留）**
- 声纹/说话人归属（属 `speaker/pipeline`）
- 唤醒词检测（KWS）与语义端点（EOC）
- 情绪/语气判别（属 `com/robot/emotion`）

---

## 2. 术语与坐标系（必须先统一）

原始描述里最大的风险是「游标」一词被同时用于两个不同含义。本设计显式引入**三个游标**，全部以「自采集启动的单调毫秒时间戳」为坐标，**不使用环形数组下标作为时间坐标**。

```
时间轴 →

        D                        C                        W
        |                        |                         |
        |<----- 待派发区 ------->|<--- 滞后区 0.5s ------->|<-- 正在写入
     已派发消费游标         VAD 已判决到此            最新提交帧
```

| 游标 | 名称 | 定义 | 推进者 |
|------|------|------|--------|
| `W` | write cursor | 最后一帧**完整提交**音频的结束时刻（不是采集线程当前时刻） | 采集线程 |
| `C` | VAD cursor | `C = W − vad_lag_ms`（默认 `lag = 500ms`） | VAD 线程 |
| `D` | dispatch cursor | 最近一次**已派发/已丢弃**段的结束时刻 | VAD 线程 |

**不变式（Assertions，代码里必须 `assert`）**

```
I1  单调序：D ≤ C ≤ W
I2  容量安全：W − D ≤ ring_capacity_ms − hard_margin_ms
I3  派发合法：任一派发段 [S,E] 满足 E ≤ C 且 S ≥ D(前值)
I4  段不重叠：D 单调不减，段与段之间只允许静音间隙
I5  SPEAKING ⇒ 存在 active_segment 且 S_active > D
```

**R7 的语义修正（关键）**
> "这段话的起点大于游标" —— 若「游标」指 `C`，则该条件不可能成立：任何被 VAD 判出的语音段，其起点 `S` 必然早于 `C`（VAD 只能看到 `C` 之前的音频）。因此这里的「游标」**只能是消费游标 `D`**，其真实意图是：**这是一段上次派发之后新出现的语音，不是旧段重复触发**。

由此 R7 拆成三个必须同时成立的独立条件：

```
R7-a  起点有效   ：S ≤ C            （VAD 已完整看过该段起点，判决可信）
R7-b  未消费     ：S ≥ D            （新段，排除重复派发）
R7-c  最短语音   ：E − S ≥ 300ms    （即 R6 的"长声音 ≥ 0.3"）
```

**三条落地约定（实现期补记，代码里都有对应断言/测试）**

1. **空闲时 `D` 跟随 `C`**：IDLE 且无候选时没有任何待派发音频，`D` 若原地不动，`W − D`
   会随静音一直涨，几秒后误触 E5 缓冲压力、把「没说的话」当溢出音频告警（TC1）。
2. **派发边界对齐到 VAD 帧栅格**（`grid0 + k·block_ms`）：`start − pre_roll` 通常落在帧中间，
   直接按毫秒切会把一个判决帧的音频劈开，回放时表现为「段首缺一块」。对齐方向：起点向下取整
   （只多给 pre_roll，不少给），终点向下取整但不早于尾点（TC3/TC7/TC15）。
3. **同句续段不得与上一段重叠**：`seq_in_utterance > 0` 的段，其 pre_roll 区间已经被上一段派发过，
   再叠一次会让下游把同 200 ms 识别两遍（表现为重复字），故续段起点直接取 `D`（TC4）。

**"长声音" 定义**：连续非静音帧构成的一段候选（candidate segment）。R6 中"没有长声音"= 当前无候选；"长声音小于 0.3"= 候选时长 < `min_speech_ms`，判为抖动/瞬态噪声，丢弃。

**"滑动窗口 2 秒" 的定位（R2 二义性，见 §7-G4）**
本设计采用**读法 A**：2 s 窗**不作为触发判决窗**，只用于帧级判决之上的统计量（噪声底噪估计、置信度平滑、双阈值自适应），窗内以帧步长连续滑动，因此**不额外引入延迟**。
读法 B（每 2 s 做一次整体判决）会把起口延迟推到 ~2.2 s，实测不可用于对话；§7-G4 给出量化对比与折中方案。

---

## 3. 数据结构

### 3.1 音频环形缓冲

```python
# com/robot/segment/ring.py

SampleRate = 16000          # 与 silero/sherpa-onnx VAD 及 SenseVoice 对齐
BytesPerSample = 2          # int16
BytesPerMs = SampleRate * BytesPerSample // 1000   # = 32 B/ms

# 6000ms → 96000 samples → 192 KB
# 建议取 2 的幂：1 << 17 samples = 131072 samples = 8192 ms（256 KB，索引可用掩码取模）
```

设计要点：
- 单写单读（SPSC）。采集线程只写 `W`，VAD 线程只读并推进 `C/D`，无锁但需 64 位原子可见（用 `threading` + 显式内存可见性约定，或 `array` + `Lock` 保底）。
- 缓冲存**原始 PCM**；帧级 VAD 概率存到独立的等长概率环（`float32`，131072 帧量级极小），避免 ASR 侧重算。
- 每帧带 `AudioFrame(samples, ts_ms, is_speech, prob, level_db, discontinuous)`，`discontinuous` 标记采集丢帧（overrun）。
- 一旦某帧落入「被 `D` 跨过且已派发」区域即可被覆盖；覆盖前必须完成深拷贝派发（见 §5.6）。

### 3.2 段 DTO

```python
# com/robot/segment/models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class SegState(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


class CloseReason(Enum):
    """段闭合原因——下游据此决定是否做同句拼接。"""
    SILENCE = "silence"            # R4：静音超时，正常闭合
    MAX_DURATION = "max_duration"  # R5：5s 硬截断，句子未说完
    RING_PRESSURE = "ring"         # I2：缓冲压力保护性闭合
    GATE_CLOSED = "gate"           # 门控关闭（唤醒超时/开始 TTS）
    EOS = "eos"                    # 采集停止/会话结束


class SampleStatus(Enum):
    """段内音频完整性。"""
    CLEAN = "clean"
    GAP = "gap"                    # 含 discontinuous 帧，可能有空洞


@dataclass
class Candidate:
    """VAD 正在累积的候选段（尚未确认）。"""
    start_ms: int
    last_speech_ms: int            # 最近一个非静音帧结束时刻
    end_ms: int                    # 当前扫描位置（≤ C）
    speech_frames: int
    silence_run_ms: int
    peak_prob: float
    peak_level_db: float
    energy_sum: float
    has_gap: bool = False


@dataclass
class Segment:
    """已闭合、可送 ASR 的整段话。"""
    seg_id: int
    utterance_id: int              # 同一句话（R5 截断后的多段共享）
    seq_in_utterance: int
    start_ms: int                  # 含 pre_roll 后的真实起点
    end_ms: int                    # 含 post_roll
    duration_ms: int
    samples: bytes                 # int16 LE 深拷贝，脱离环形缓冲
    sample_rate: int = 16000
    channels: int = 1
    close_reason: CloseReason = CloseReason.SILENCE
    status: SampleStatus = SampleStatus.CLEAN
    vad_confidence: float = 0.0    # 段内 speech 帧占比
    noise_floor_db: float = -60.0
    partial: bool = False          # §5.7：True=提前送出的临时段，需 finalize
    meta: dict = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return not self.partial
```

---

## 4. 参数表（集中配置，单位统一 ms / dB）

```python
# com/robot/segment/config.py（与落库实现一致）

@dataclass
class SegmentConfig:
    # ---- 采集 ----
    sample_rate: int = 16000
    channels: int = 1                  # 仅 mono（见 §7-G18）
    block_ms: int = 32                 # 采集块 = VAD 帧（512 样本 @16k）
    ring_capacity_ms: int = 8192       # ⚠ 6000 在本参数集下不成立，见 §7-G5

    # ---- 游标 ----
    vad_lag_ms: int = 500              # R1；且必须 ≥ 批量型 VAD 的 lag_ms（见 §7-G19）
    hard_margin_ms: int = 500          # I2 保护带

    # ---- 判决窗（R2 读法 A：统计窗，不用于触发；按帧连续滑动）----
    stat_window_ms: int = 2000

    # ---- 帧级阈值（双阈值迟滞，见 §7-G13 的实现口径）----
    prob_start: float = 0.55           # IDLE 侧：p ≥ 此值才可能起口
    prob_end: float = 0.35             # SPEAKING 侧：p < 此值才算静音（中间带不推进静音计时）
    confirm_frames: int = 6            # 连续 6 帧（~192ms）过 prob_start 才认定 onset

    # ---- 段长度约束 ----
    min_speech_ms: int = 300           # R6 / R7-c
    min_silence_ms: int = 300          # R4（建议 500~650，见 §7-G1）
    long_silence_ms: int = 650         # 二次静音：确认句末（仅 emit_partial 模式生效，见 §7-G2）
    gap_merge_ms: int = 200            # 短间隙并入同段（见 §7-G6）
    gap_force_close_ms: int = 200      # 采集空洞 ≥ 此值 → 强制闭合（见 §7-G12）
    max_utterance_ms: int = 5000       # R5
    cut_search_ms: int = 500           # 截断点择优的回溯窗（见 §5.5）
    pre_roll_ms: int = 200             # 起点向前扩（见 §7-G7）
    post_roll_ms: int = 100

    # ---- 能量门限（见 §7-G10 / §7-G20）----
    dispatch_min_level_db: float = -45.0
    min_level_over_noise_db: float = 6.0

    # ---- 派发 ----
    emit_partial: bool = False         # False = batch（R7 原口径），True = 流式提前送
    # 队列容量不在这里：QueueDispatcher(maxsize=4)，见 §5.6

    # ---- 启动 / 退出边界（见 §7-G15）----
    warmup_ms: int = 500

    # ---- 门控默认值（见 §7-G8 / §7-G9）----
    gate_mode: str = "half"            # half | full（full 需 AEC）；armed/tts_playing 是运行期状态，
    ignore_during_tts: bool = True     # 归 StaticGate，不放配置里
```

> 实现说明：本文早期草稿里写过的 `partial_at_ms`、`armed`、`tts_playing` 三个字段最终没有落进
> `SegmentConfig`——partial 的触发时刻就是 `min_silence_ms`、final 的确认时刻就是 `long_silence_ms`，
> 再加一个 `partial_at_ms` 只会造出第三处互相矛盾的静音参数；`armed/tts_playing` 是运行期状态，
> 由 `StaticGate` 持有（门控与主状态正交，见 §5.1）。`stat_stride_ms` 同样被删掉：统计窗按帧滑动，
> 步长恒等于 `block_ms`，写成可配参数等于给人一个改了没用的旋钮。

**容量约束（必须作为断言/CI 检查存在）**

```
ring_capacity_ms ≥ max_utterance_ms + vad_lag_ms + pre_roll_ms + post_roll_ms + hard_margin_ms

代入默认：5000 + 500 + 200 + 100 + 500 = 6300 ms  >  6000 ms   ← 不满足
```

原始 6 s 缓冲**在本参数集下不成立**。三个解法（择一，文档默认取①）：
① `ring_capacity_ms = 8000`（192 KB → 256 KB，代价可忽略）；
② `max_utterance_ms = 4200`；
③ 取消 `hard_margin`、靠溢出强制闭合兜底（不推荐，会随机丢字）。

---

## 5. 状态机设计

### 5.1 状态图（保持"两状态"要求）

```
                        ┌──────────────────────────────────────────┐
                        │                                          │
                        ▼                                          │
   ┌──────────┐  E2 onset(R7-a/b/c) + R5/R4 gate   ┌───────────┐   │
   │   IDLE   │───────────────────────────────────►│ SPEAKING  │   │
   │ (R3 初值)│                                     │           │   │
   └──────────┘ ◄────────────────────────────────── └───────────┘   │
        ▲    E3 offset(R4) / E4 max_dur(R5) / E5 ring(I2) / E6 gate │
        │                                                           │
        └──── E1 candidate < min_speech(R6) → discard ──────────────┘
              （未确认候选直接丢弃，主状态不变，仍为 IDLE）

  自环：IDLE 自环 = R4「静音>0.3s 仍为空闲」（无动作，仅复位计时器）
```

> 主状态只有 2 个，符合原始约束。**但必须存在一个正交的 `pending_flush` 标志**（§5.6）：状态可以已经回到 IDLE，而段的深拷贝与投递尚未完成。否则"切状态"与"送数据"在同一线程里串行，会阻塞 VAD 扫描。若希望更强的可读性，可升级为第三态 `FINALIZING`（§7-G3 给出理由），本设计默认不升级，以保持与原始约束一致。

### 5.2 事件定义

| 事件 | 触发条件 | 来源 |
|------|----------|------|
| `E0_FRAME(p, t)` | VAD 处理一帧，`t ≤ C` | 帧循环 |
| `E1_INVALID_CAND` | 候选存在但 `dur < min_speech_ms` | E0 派生 |
| `E2_ONSET` | 连续 `confirm_frames` 帧 ≥ `prob_start`，且 `S ≤ C`、`S ≥ D`、`E−S ≥ min_speech_ms` | E0 派生 |
| `E3_OFFSET` | `C − last_speech_ms ≥ min_silence_ms` | 游标推进 |
| `E4_MAX_DUR` | `C − S ≥ max_utterance_ms` | 游标推进 |
| `E5_RING_PRESSURE` | `W − D ≥ ring_capacity − hard_margin` | I2 监视 |
| `E6_GATE` | `armed=False` 或 `tts_playing=True`（半双工时） | 外部 |
| `E7_GAP` | 采集 discontinuous 帧 | 采集线程 |

### 5.3 转移表（含守卫与动作）

| # | 当前态 | 事件 | 守卫 | 下一态 | 动作 |
|---|--------|------|------|--------|------|
| T1 | IDLE | E0 | — | IDLE | 更新统计窗、噪声底噪；累积候选 |
| T2 | IDLE | E2 | R7-a ∧ R7-b ∧ R7-c ∧ level > `dispatch_min_level_db` | SPEAKING | `S = onset_ts − pre_roll`；建 `Candidate`；`utterance_id++`；fire `on_speech_start` |
| T3 | IDLE | E1 | `dur < min_speech_ms` | IDLE | **丢弃候选但推进 `D = E`**（防重复触发）；`false_trigger++` |
| T4 | IDLE | E3 | — | IDLE | 复位 `silence_run`（R4 的自环，语义为"确认空闲"） |
| T5 | SPEAKING | E0 | — | SPEAKING | 累积 speech 帧；若间隙 < `gap_merge_ms` 不结束候选 |
| T6 | SPEAKING | E3 | `C − last_speech ≥ min_silence_ms` | IDLE | 闭合段（`CloseReason.SILENCE`）→ `pending_flush`；`D = E`；fire `on_segment` |
| T7 | SPEAKING | E4 | `C − S ≥ max_utterance_ms` | **IDLE**（R5 要求） | 闭合段（`MAX_DURATION`）；**截断点择优**（§5.5）；同 `utterance_id`，`seq+1`；若此刻仍在说话，下一帧可立即重入 SPEAKING |
| T8 | SPEAKING | E5 | I2 违例前兆 | IDLE | 强制闭合（`RING_PRESSURE`）；记 `overflow_risk++`；触发参数告警 |
| T9 | SPEAKING | E6 | 门控关闭 | IDLE | 若 `dur ≥ min_speech_ms` 仍闭合派发（`GATE_CLOSED`），否则丢弃 |
| T10 | SPEAKING | E7 | 出现空洞 | SPEAKING | 标记 `status=GAP`；空洞 > 200ms 则强制闭合 |
| T11 | * | E7 | 空洞且未 SPEAKING | IDLE | `D` 跳过空洞区，避免时间轴撕裂 |
| T12 | * | E1/E2 同时 E3/E4 | 优先级见 §5.4 | — | — |

### 5.4 事件优先级（解决"规则竞态"）

原始规则未定义同时满足时的顺序。固定优先级：

```
E5_RING_PRESSURE  >  E4_MAX_DUR  >  E3_OFFSET  >  E6_GATE  >  E2_ONSET  >  E1_INVALID_CAND
```

理由：E5 是数据安全问题（不处理会丢音频）；E4/E3 都闭合段，若在同一扫描点同时成立，说明静音恰好在 5 s 边界出现，取 E4 并让 `CloseReason` 记录两者（`MAX_DURATION` + `meta["also_silence"]=True`），避免下游重复拼接。

### 5.5 与 R5（5 s 硬截断）相关的补全

R5 只说"截断，状态机为空闲"，未定义三件事，必须补齐：

1. **截断点择优**：不要在 `S + 5000ms` 处硬切（大概率切在元音上，ASR 出脏字）。在 `[S+4500, S+5000]` 区间内选 **能量最低的帧边界**（且 `prob < prob_end`）作截断点；找不到则取区间起点。
2. **截断后是否立即重入 SPEAKING**：R5 置 IDLE 后，若人还在说，下一帧会满足 E2 并重入。必须保证 `S_new > 截断点` 且 `S_new ≥ D`，否则重复派发。文档以 `D = 截断点` 保证。
3. **句子归属**：截断说明句子没说完，必须用 `utterance_id` 让下游把多段**拼回同一句**（R7 的"一整段话"目标不因硬切而失效）。同时建议 `max_utterance_ms` 触发时先发 `partial=True`，再由静音/后续段 finalize。

### 5.6 段派发（R7 后半句"将一整段话送入后续处理"）

```
闭合 ──► 深拷贝 ring[S:E] ──► 立即推进 D ──► 投递队列（非阻塞）──► ASR 线程
```

要点：
- **D 在拷贝完成后立即前进，绝不等待 ASR 结果**。否则慢 ASR 会把 `D` 拖住，`W − D` 撑爆环形缓冲（§7-G11）。
- 投递队列有界（`QueueDispatcher(maxsize=…)`，装配时指定，不属于状态机参数）。满时**只挤掉最旧的 partial**、
  final 从不主动丢弃，并记 `backpressure_drop++`（TC11b 专门验证「丢 partial 保 final」）。
- `Segment.samples` 是脱离环形缓冲的 `bytes`，下游可安全持有。
- 派发后保留帧级元数据（`start/end/prob/level`）供 `emotion`、`memory` 与调试使用。

接口对齐仓库既有风格（`docs/speaker_pipeline_design.md` 的 ABC 注入方式）：

```python
class ISegmentDispatcher(ABC):
    @abstractmethod
    def dispatch(self, seg: Segment) -> bool: ...   # 非阻塞，False = 队列满

class ISpeechGate(ABC):
    """唤醒 / TTS 自听 / 半双工门控（§7-G8、§7-G9）。"""
    @abstractmethod
    def allow(self) -> Tuple[bool, str]: ...
```

### 5.7 一次性整段 vs 流式 partial（R7 的延迟代价）

R7 要求"一整段话送入后续处理"，即 **batch 模式**：段闭合前一个字节都不送 ASR。代价是首字延迟 = 段长 + 闭合延迟。对话场景建议提供两档；**落库默认 batch**（严格照 R7 口径，且不引入重复计算），需要快速反馈时显式开 `emit_partial=True`：

```
min_silence_ms 到达 → 送 partial（先出结果，用于快速响应/打断判断）
long_silence_ms 到达 → 送 final（同 seg_id，下游以 final 覆盖 partial）
```

`Segment.partial` + `CloseReason` 已覆盖此语义。SenseVoice 离线模式下，`partial` 会重复计算，故**只在需要快速反馈时开启**；纯离线 ASR 走 batch。
两条实现细节（§8 TC11b/TC13 覆盖）：partial **不推进 `D`**（final 会以更宽区间重读并覆盖同一 `seg_id`）；
发出 partial 后若用户又接着说，状态机必须**留在 SPEAKING**，`long_silence_ms` 才是真正的句末。

### 5.8 参考实现（帧循环骨架）

> 下面是**设计期骨架**，用于对照 §5.3 转移表；落库实现见 `com/robot/segment/engine.py`，
> 与骨架有四处有意的差别：
> ① `speech` 按状态取阈值（`IDLE` 用 `prob_start`、`SPEAKING` 用 `prob_end`，§7-G13）；
> ② 事件顺序严格按 §5.4（`E5 > E4 > E3 > E6 > E2 > E1`），骨架里放在最后的 `RING_PRESSURE`
> 实际上在函数入口先判；
> ③ 派发边界对齐帧栅格、同句续段不重叠、空闲时 `D` 跟随 `C`（§2 三条落地约定）；
> ④ 截断后不是直接丢弃剩余音频，而是用 `cut` 之后的帧重建候选并归入同一 `utterance_id`（§5.5）。

```python
# com/robot/segment/engine.py（骨架，逻辑与 §5.3 转移表一一对应）
def step(self, frame: AudioFrame) -> None:
    """由 VAD 线程按帧驱动；frame.end_ms <= C。"""
    self._w = frame.end_ms + self.cfg.vad_lag_ms          # 采集侧写入的 W 快照
    self._c = frame.end_ms                                 # 游标推进

    if self.cfg.stat_window_ms > 0:                        # R2：统计窗，不用于触发
        self._stats.push(frame)
        self._noise_floor = self._stats.percentile_db(20)  # 自适应底噪

    speech = frame.prob >= self.cfg.prob_start
    quiet = frame.prob < self.cfg.prob_end

    # ---- 空洞（T10/T11）----
    if frame.discontinuous:
        self._on_gap(frame)

    # ---- IDLE：候选累积与 onset（T1/T2/T3/T4）----
    if self._state is SegState.IDLE:
        if speech:
            self._cand.extend(frame)                       # 候选只由 VAD 帧构成
        elif self._cand and frame.start_ms - self._cand.last_speech_ms <= self.cfg.gap_merge_ms:
            pass                                           # T5 前置：短间隙不结束候选
        elif self._cand:
            dur = self._cand.last_speech_ms - self._cand.start_ms
            if dur >= self.cfg.min_speech_ms and frame.start_ms - self._cand.last_speech_ms >= self.cfg.min_silence_ms:
                self._emit_close(self._cand, CloseReason.SILENCE)   # 起口即收尾的极短段
            else:
                self._discard_cand()                        # T3：R6，推进 D，false_trigger++
        if self._cand and self._cand.confirm_ready():       # E2
            ok, why = self._gate.allow()
            if not ok:
                self._discard_cand(reason=why)              # T9
            else:
                self._to_speaking(self._cand)               # T2 → SPEAKING

    # ---- SPEAKING（T5/T6/T7/T8）----
    else:
        if speech:
            self._act.extend(frame)
        if self._c - self._act.last_speech_ms >= self.cfg.max_utterance_ms - self._act.duration_ms:
            self._emit_close(self._act, CloseReason.MAX_DURATION)  # T7：R5
        elif self._c - self._act.last_speech_ms >= self.cfg.min_silence_ms:
            self._emit_close(self._act, CloseReason.SILENCE)       # T6：R4
        if self._w - self._d >= self.cfg.ring_capacity_ms - self.cfg.hard_margin_ms:
            self._emit_close(self._act, CloseReason.RING_PRESSURE)  # T8：I2
```

> 注：实现时把 `E4` 的守卫写成 `C − S ≥ max_utterance_ms` 更直观；上面为演示"与 T6 的优先级"写成等价形式，落库时统一为 §5.2 的定义并按 §5.4 排序。

---

## 6. 时序与延迟预算

```
帧级判决粒度      = block_ms                  = 32 ms
起口延迟(onset)   = vad_lag + ceil(min_speech / block) × block
                  =   500   +      ceil(300/32)×32 = 320  = 820 ms
闭合延迟(offset)  = vad_lag + ceil(min_silence / block) × block = 500 + 320 = 820 ms
                  （emit_partial 时 final 用 long_silence=650 → 672 → 1172 ms；partial 仍 820 ms）
硬截断延迟        = vad_lag + max_utterance             = 500 + 5000 = 5500 ms（上界）
端到端(闭→文本)   = offset 820 + ASR(SenseVoice RTF≈0.1 × 段长)
                  段长 3s → ≈ 1.1 s；段长 5s → ≈ 1.3 s
```

`SegmentConfig.latency_budget()` 直接返回上面三个数（820 / 820 / 5500），CI 用例
`test_latency_budget_matches_doc` 把它们钉死，改参数时文档与代码不会各说各话。

> 订正：早期版本把 onset 写成 `vad_lag + min_speech + confirm_frames×block ≈ 992 ms`，这里
> `confirm_frames` 被重复计算了——`confirm_frames×block = 192 ms` 的连续起口帧**本来就在**
> `min_speech = 300 ms` 的窗口里数完，不是串联关系。只有当 `confirm_frames×block_ms > min_speech_ms`
> 时起口才由确认帧决定，而那已经是配错参数（`validate()` 会直接拒绝，见 §4 与 TC2 的守卫）。
> 实测值见 §8：TC7 的 onset = 320 ms（未叠加 lag 的逐帧驱动），`latency_budget()` 即按上式返回。

结论：
- 起口反馈 1 s、出文本 ~1.3 s，对"陪伴机器人"**可接受但不宽裕**。若要更敏捷，唯一有效手段是 §5.7 的 partial（可把出文本推到 ~0.6 s），而不是压缩 `min_silence`（会更碎）。
- 若采用 R2 读法 B（2 s 触发窗），起口延迟变为 `lag + window − min_speech ≈ 2.2 s`，对话体验直接崩坏 —— 这是 §7-G4 的核心结论。

---

## 7. 原始描述的不足清单（逐条：问题 → 影响 → 补全）

### G1 — "静音大于 0.3 秒"对中文过短，会把一句话切碎【高】
中文句内停顿（并列、思考、换气）常在 0.3–0.6 s。0.3 s 即判句末 → 大量碎片段 → ASR 出残句、下游语义不连贯、`memory` 记不到完整事件。
**补全**：两级静音。`min_silence_ms=300` 只发 `partial`，`long_silence_ms=650` 才发 `final`；并在闭合处检查"是否以逗号/未完整语气结尾"（由 ASR 标点后判，接口预留 `Segment.meta["needs_stitch"]`）。碎片仍用 `utterance_id` 拼接。

### G2 — 未定义"句末"与"句中停顿"的区分【高】
同 G1。R4/R7 只给了时长维度，缺语义/能量趋势维度。
**补全**：闭合决策加入三个辅助特征（均在 2 s 统计窗内可得）：段末 `level_db` 相对段峰值的下降斜率、末帧 `prob` 是否已跌破 `prob_end`、`silence_run` 是否跨过一个完整帧块。三者加权后作 threshold，而非单一时间量。

### G3 — "转空闲"与"送后续处理"是两个动作，但只有一个状态位【中】
状态立刻回 IDLE 而拷贝/投递未完成时，若紧接着来新段会出现 `D` 未推进 → 重复派发或派发乱序。
**补全**：显式 `pending_flush` 标志 + 独立派发线程；或接受升级为 `FINALIZING` 第三态（§5.1 已给保守方案）。文档默认前者，以保持"两状态"约束。

### G4 — R2「滑动窗口 2 秒」语义未定义，两种读法延迟差 1 秒以上【高】
读法 A（统计/平滑窗）→ 零额外延迟；读法 B（每 2 s 整体判决）→ 起口延迟 ~2.2 s。
**补全**：明确取 A，统计窗按帧（`block_ms`）连续滑动即可，不需要独立步长参数（早期草案里的 `stat_stride_ms` 已删——它等于给了一个改了没用的旋钮）。若坚持 B 的抗误触发能力，折中为「帧级触发 + 2 s 窗二次确认」，但二次确认必须与触发并行、不得阻塞 onset。

### G5 — 6 s 缓冲与 5 s 上限 + 0.5 s 滞后的余量算紧【高，量化】
`5000 + 500 + 200 + 100 + 500 = 6300 > 6000`。§4 断言不成立。
**补全**：容量提到 8 s（1 `<< 17` 样本 = 8192 ms，掩码取模，多 64 KB）；或上限降到 4.2 s。并把该不等式写成启动期 `assert` 与 CI 检查，防止后续调参悄悄破约束。

### G6 — 未定义段内短间隙（gap merge）【中】
R6/R7 以"连续非静音"定义段，但 VAD 帧概率必然有瞬时跌破，会把一个词内轻音切成两段。
**补全**：`gap_merge_ms = 200`：间隙 < 此值不结束候选，段仍视为连续；`min_silence_ms` 只在间隙 ≥ 该值后开始计时。

### G7 — 缺 pre-roll，起始辅音/vowel onset 会被吃掉【高】
VAD 需要 `confirm_frames` 才认定起口，段起点若取"判决时刻"，实际语音前 ~200 ms 已丢失，表现为首字缺声、ASR 漏字。
**补全**：`pre_roll_ms = 200`、`post_roll_ms = 100`，派发时从环形缓冲的历史区间取。
**订正（实现期）**：pre-roll 的来源是**环形缓冲保留的历史深度**（`capacity ≫ pre_roll`，且约束不等式已保证
`W − D ≤ capacity`），与 `vad_lag` 无关——本文早期版本写作"R1 的 0.5 s 滞后正好为此提供空间"，该论断不成立，
已在 §2 落地约定中改正。`vad_lag_ms` 的真实作用是两条：① 只对**已完整提交**的帧做判决（I1 的前提）；
② 覆盖批量型 VAD 的判决步进（见 §7-G19）。pre-roll 必须 ≤ 缓冲保留长度，否则读出 `CLAMPED` 补零音频。

### G8 — 没有唤醒门控，与仓库既定架构脱节【高】
`readme.md` 明确"唤醒方案[预制关键词，自动唤醒]"，本状态机未接入。空闲态若一直派发，`memory`/LLM 会被环境噪声持续灌入。
**补全**：`ISpeechGate` + `armed` 标志（与主状态**正交**，保持两状态）。未 armed 时：不派发、`D` 照常推进、仅维持 VAD 统计供 KWS 使用。armed 的超时（如 30 s 无交互）自动 disarm。

### G9 — 没有自听/AEC 处理，机器人会被自己的 TTS 触发【高，实现期必现】
扬声器播放 = mic 必录到 → 立刻 E2 onset → 状态机转 SPEAKING → 把机器人自己的话当用户输入，形成自问自答。这是语音机器人最常见的失效模式。
**补全**：三档策略，按硬件能力选一：
① `half`：`tts_playing=True` 期间 gate 关闭，`D` 快进到播放结束（最简单，牺牲打断能力）；
② `full`：接 AEC（WebRTC AEC3 / RK 平台对应处理）后再进 VAD，允许 barge-in，且 onset 需 AEC 后残差能量达标；
③ 折中：播放期间提高 `prob_start`（0.55→0.75）并要求 `confirm_frames` 加倍。
`gate_mode` 已预留。

### G10 — 缺能量/信噪门限，纯时长判据不够【中】
R6/R7 只用"时长 > 0.3 s"。远场空调、键盘、翻页都可能是持续非静音。
**补全**：`dispatch_min_level_db = −45`（相对 2 s 窗自适应底噪 `noise_floor_db + 6 dB` 取大者），双门限（时长 ∧ 能量）才 onset。

### G11 — 未定义消费端慢/失败时的行为（背压与 `D` 的推进时机）【高】
若 `D` 等 ASR 返回才前进，环形缓冲必然溢出；若不等，ASR 失败无法重试（音频已被覆盖）。
**补全**：`D` 在**深拷贝后**立即前进（§5.6）；投递队列有界 + 丢弃策略 + 计数；重试只能重放 `Segment` 对象，不能重放环形缓冲。补一条降级：ASR 队列深度 > 阈值时**主动丢弃 partial、只保 final**，并上报"响应降级"事件（情绪/调度模块可据此降低主动发言频率）。

### G12 — 未定义采集丢帧（overrun / xrun / 设备重启）【中】
readme 记录的环境是 WSL + pulseaudio 桥接 + pyaudio，overrun 概率不低。音频出现空洞时 R7 的"一整段话"不再完整。
**补全**：帧带 `discontinuous`；空洞 < 200 ms 用静音填充并标 `status=GAP`；≥ 200 ms 强制闭合（`CloseReason.EOS` 语义）；空洞计入 `drop_frames_ms` 指标并触发设备健康告警。

### G13 — 缺帧级双阈值与迟滞，状态机会在阈值附近抖动【中】
单一阈值 + 单帧判 → 反复 IDLE↔SPEAKING。
**补全**：`prob_start / prob_end` 双阈值（§4）+ `confirm_frames` + `gap_merge`。文档同时建议：概率序列进独立环形缓冲，便于回放与阈值调参。
**实现口径（重要，容易写错）**：迟滞按**状态**取阈值，而不是所有帧用一个阈值——
`IDLE` 侧起口要求 `prob ≥ prob_start`（进 SPEAKING 要难），
`SPEAKING` 侧只有 `prob < prob_end` 才算静音（收尾要松），中间带既不另起新段也不推进静音计时（TC21）。
另有一条约束：VAD 后端自身的时域平滑（能量 VAD 的 release 拖尾）必须**短于 2 帧**。
若后端拖 100 ms、状态机再等 300 ms，两处判决叠加就等于 400 ms 静音，句间 400 ms 的正常停顿会被并成一句
（`com/robot/models/asr/asr.py` 的「0.8~1.3 s 才出文字」就是同一类叠加，见 §7-G19）。

### G14 — 未定义"起点大于游标"的真实所指【高，语义】
若按 `C` 理解则 R7 永假（§2 已推导）。
**补全**：确认按 `D` 理解，并把 R7 拆成 R7-a/b/c 三条显式守卫。

### G15 — 缺启动/退出边界处理【低】
R3 只说"开始启动为空闲"。mic 上电 pop 音、首次噪声底噪未估计时会误触发。
**补全**：`warmup_ms = 500` 丢弃；`noise_floor` 未收敛前 `dispatch_min_level_db` 保守提高；采集停止时以 `CloseReason.EOS` 闭合在途段（否则最后半句丢失）。

### G16 — 缺可观测性与验收标准【中】
设计里没有指标，调参没有依据；G1/G10/G12 的"是否更好"无法判定。
**补全**：§8。

### G17 — 与 `speaker/pipeline` 的职责边界未界定【中】
`docs/speaker_pipeline_design.md` 已定义 `IAudioPreprocessor`（分帧 + VAD + 语音段提取）与 `IASREngine`。本模块与它**功能重叠**。
**补全**：明确分层——本 `segment` 模块是**实时双状态机（时间轴驱动，产出流式段）**；`IAudioPreprocessor.extract_speech_segments` 是**批式接口（整文件输入）**。二者共享帧级 VAD 与 `AudioSegment` DTO，但**不共享状态机**。落库时 `com/robot/segment/models.py` 复用 `pipeline/models.py` 的 `AudioFrame/AudioSegment`，避免两套数据结构漂移。

### G18 — 单通道假设未声明，与多说话人流水线的衔接缺失【低】
本状态机产出的是**混合音频段**（单通道）。若后续接 DOA/波束形成，段应是"每波束一路"，则状态机需实例化为 M 路。
**补全**：`SegmentEngine` 设计为**每通道/每波束一个实例**（无内部全局态），由上层 `SpeakerPipeline` 按 `List[Beam]` 扇出；本文档默认 mono 实例。

### G19 — 没有区分两类 VAD 后端，帧概率型与段输出型的延迟口径完全不同【高，实现期必踩】
状态机的输入是「逐帧语音概率」，但可用后端分两型：
① **帧概率型**（silero / 能量）：每帧一个 `[0,1]`，与帧时钟严格同步，闭合延迟就等于 `vad_lag + min_silence`；
② **段输出型**（FunASR `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`）：一次输入整窗音频，吐**语音段毫秒区间列表**
（`value=[[start_ms, end_ms]]`，单位是毫秒——`asr.py` 里专门标注过这个坑），它**不给逐帧概率**。

不区分就会出现三件事：
1. **模型自带尾静音与状态机静音阈值叠加**。FSMN 默认 `max_end_silence_time ≈ 800 ms` 才吐段尾，
   状态机再等 `min_silence_ms = 300`，真实闭合延迟约 0.8~1.3 s——这正是 `asr.py` 尾注里
   「说完后约 0.8~1.3 秒出文字」的成因，而不是 ASR 慢。适配器把该值显式记录为
   `WindowSegmentVad.model_end_silence_ms`，调参时必须知道自己是**在它之外又加了一层**。
2. **批量判决的固有滞后**。每 `stride` 重算一次整窗，就只能每 `stride` 交付一批帧概率，
   第 `i` 帧拿到的是 `i − stride + block` 的判决，滞后恒为 `lag_ms = stride_ms − block_ms`
   （默认 512−32 = 480 ms）。**`vad_lag_ms` 必须 ≥ `lag_ms`**，否则状态机以为的帧时间和概率实际描述的帧不是同一帧，
   pre/post_roll 与段边界会整体偏移。这一条正好解释了需求里「VAD 游标在写指针前 0.5 秒」与
   `asr.py` 里 `VAD_STEP_MS = 500` 为什么是同一个数。
3. **窗尾贴段尾 ≠ 说完了**。段输出型在窗尾仍处语音时必须把窗尾若干帧判为语音（`ongoing_margin_ms`），
   否则每滑一个窗都会看到一次假收尾（TC20）。

**补全**：`ISegmentVAD.prob(samples) -> float` 统一两型；`WindowSegmentVad` 负责段列表 → 逐帧概率的换算，
并暴露 `model_calls`（重算次数）与 `lag_ms`（固有滞后）供观测与 §4 不等式校验；冗余计算比 = `window_ms / stride_ms`
（默认 1984/512 ≈ 3.9×，优于 `asr.py` 的 2.5 s 窗 / 0.5 s 步进 = 5×）。真要再降延迟，只能换流式 VAD，
调 `min_silence_ms` 无效——那层等待在模型里面。

### G20 — 自适应底噪若把语音帧算进去，连续说话会被自己的电平判成「能量不足」而整段丢弃【高，实测踩到】
`dispatch_min_level_db` 的自适应项按窗内分位数估计底噪时，若窗内几乎全是语音（例如 5 s 上限内一直在说），
20 分位数就等于语音电平，门限变成「语音电平 + 6 dB」→ 永远不达标 → 截断后的续段直接消失（TC4 当时就挂在这里）。
**补全**：底噪**只用 `prob < 0.5` 的非语音帧**估计；窗内非语音帧少于 8 帧视为未收敛，返回默认极小值，
此时门限退化为静态 `dispatch_min_level_db`（宁可用静态门限，也不要用被语音污染的自适应值）（TC14b）。

---

## 8. 可观测性与验收

**指标（每 1 s 汇总 + 会话结束输出）**

| 指标 | 定义 | 目标 |
|------|------|------|
| `onset_latency_ms` | onset 判决时刻 − 真实起口（离线标注） | P50 ≤ 1000, P95 ≤ 1300 |
| `offset_latency_ms` | 段派发时刻 − 真实收尾 | P50 ≤ 850 |
| `fragmentation_rate` | `dur < 1.5×min_speech` 的 final 段占比 | < 8% |
| `false_trigger_rate` | 空文本段 / 总段数 | < 5% |
| `max_dur_hit` | `CloseReason.MAX_DURATION` 次数 | < 10% 段数 |
| `ring_pressure` | `CloseReason.RING_PRESSURE` 次数 | **0** |
| `backpressure_drop` | 投递丢弃数 | ≈ 0 |
| `drop_frames_ms` | 空洞累计时长 | < 0.5% 会话时长 |
| `e2e_to_text_ms` | 段闭合 → ASR 首 token | P50 ≤ 1200 |

**Golden 测试用例（离线 wav + 时间轴标注，逐帧驱动 `step()`）**

| # | 输入 | 期望 |
|---|------|------|
| TC1 | 纯静音 10 s | 全程 IDLE，0 段（R4） |
| TC2 | 0.2 s 瞬态（拍手） | 候选丢弃，`D` 前进，0 段（R6/T3） |
| TC3 | 1.0 s 语音 + 1.0 s 静音 | 1 段，`SILENCE`，`S = onset−200`，`E = last_speech+100`（R4/R7） |
| TC4 | 6.0 s 连续语音 | 至少 2 段，首段 `MAX_DURATION`，**同 `utterance_id`**，截断点非语音峰值（R5 + §5.5） |
| TC5 | 语音中 0.15 s 间隙后继续 | **1 段**，非 2 段（G6） |
| TC6 | 语音中 0.4 s 停顿后继续 | 2 段同 `utterance_id`（G1/G2） |
| TC7 | 起口 0.1 s 内即 onset | 段起点必须早于 onset 帧 200 ms（G7） |
| TC8 | onset 与 max_dur 同帧成立 | 取 `MAX_DURATION`，`meta["also_silence"]`（§5.4） |
| TC9 | TTS 播放期间灌入机器人自身语音 | 0 段（G9，half 模式） |
| TC10 | 中途注入 300 ms 空洞 | 段闭合 + `status=GAP`（G12） |
| TC11 | 消费队列塞满 | `D` 仍前进、无异常、`backpressure_drop++`（G11） |
| TC12 | `ring_capacity=6000, max_utterance=5000` | 启动 `assert` 失败（G5） |

实现期新增（同一文件内，逐条对应上面「补全」里立的规矩；`test_segment.py` 共 37 个用例）：

| # | 输入 | 期望 | 覆盖 |
|---|------|------|------|
| TC1 补充 | 纯静音 10 s | `D == C`（空闲水位线），且 `RING_PRESSURE == 0` | §2 约定 1 |
| TC9b/c | 未 armed / `full` 模式 + TTS 播放 | 前者 0 段且 `gate_blocked` 只在开→关计一次；后者照常派发 | G8/G9 |
| TC11b | 队列 1 格 + `emit_partial` | 被挤掉的是 partial，final 留下 | G11 |
| TC13 | `emit_partial` 下停顿后又接话说 | 同 `seg_id`，`final.end > partial.end`，之后离开 SPEAKING | §5.7 |
| TC14/14b | 远场低电平语音 / 窗内几乎全是语音 | 前者被能量门限拦下并 discard；后者底噪仍为 −70 dB 不误伤 | G10/G20 |
| TC15 | `ring=2048ms` 反复说 4 轮（必然回绕） | 每段 `status=CLEAN` 且首样本时刻 == `start_ms` | §3.1 |
| TC16 | 混合长序列 | 全程 `D ≤ C ≤ W`、`W − D ≤ capacity`（step 内 assert） | I1/I2 |
| TC17 | 说话中停止采集 | `flush()` 以 `EOS` 闭合在途段 | G15 |
| TC18 | `feed()` 全链路（PCM → 能量 VAD → 状态机） | 切出 1 段，边界包住真实说话区间 | §5.8 |
| TC19 | 段输出型 VAD 适配 | 每帧恰好一个概率，滞后恒为 `stride − block`，模型调用次数 == 批数 | G19 |
| TC20 | 段尾贴窗 vs 段尾远离窗尾 | 前者窗尾必须为语音，后者必须落到静音 | G19 |
| TC21 | `SPEAKING` 中跌进 `[prob_end, prob_start)` 中间带 | 不提前闭合；`IDLE` 侧中间带不起口 | G13 |

**调试回放**：帧级 `prob` + 判决 + 段边界落一行 JSONL（`ts, prob, state, action, cand_id`），配合一段 wav 可完整复现任何误触发；`_debug_*.py` 的现有风格可沿用（参考 `com/robot/scheduler/_debug_wheel.py`）。

---

## 9. 落库结构（对齐仓库现有命名）

```
com/robot/segment/
├── __init__.py          # re-export SegState / Segment / SegmentEngine / SegmentConfig
├── config.py            # SegmentConfig + §4 容量断言 + latency_budget()
├── models.py            # SegState / CloseReason / SampleStatus / AudioFrame / Candidate / Segment
├── ring.py              # AudioRingBuffer（SPSC，2 的幂掩码，越界显式报 CLAMPED/dropped）
├── vad.py               # ISegmentVAD：EnergyVad / ScriptedVad / SherpaSileroVad / WindowSegmentVad
├── engine.py            # SegmentEngine：§5.3 转移表的状态机（可逐帧 step() 驱动）
├── dispatcher.py        # ISegmentDispatcher：QueueDispatcher（有界 + 丢 partial 保 final）/ CallbackDispatcher
├── gate.py              # ISpeechGate：armed / tts_playing / gate_mode
├── stats.py             # §8 指标聚合 + 非语音帧底噪估计（G20）
├── demo.py              # 离线回放：wav → 分段 → 落盘段文件 + 时间轴报告
├── demo2.py             # 实时采集：麦克风(1/多通道) → 重采样/下混 → 分段
└── test_segment.py      # TC1–TC21（37 个用例，纯标准库）

docs/segment_pipeline_design.md   # 本文档
```

核心逻辑只依赖标准库（`dataclasses/abc/array/math/queue/threading/logging`），
`numpy / funasr / sherpa_onnx / sounddevice` 全部惰性导入，因此状态机在 RK 板子和 CI 上都能直接跑：

```bash
python -m unittest com.robot.segment.test_segment -v        # 状态机与后端全链路
python -m com.robot.segment.demo --vad energy in_16k_mono.wav          # 时间轴报告
python -m com.robot.segment.demo --vad fsmn --out /tmp/seg in_16k_mono.wav   # 需要 funasr

python -m com.robot.segment.demo2 --list-devices                       # 看可用麦克风与通道数
python -m com.robot.segment.demo2                                      # 默认设备实时分段（需 sounddevice）
python -m com.robot.segment.demo2 --channels 2 --mix energy --out /tmp/seg
python -m com.robot.segment.demo2 --channels max --mix multi --duration 30
python -m com.robot.segment.demo2 --input sim --channels 4 --mix multi  # 无麦克风环境自测
```

### 9.2 采集侧参数如何落到本文的时间坐标

`SegmentConfig` 里所有判决门限都是**毫秒**，而设备给的是**样本数**，二者之间只靠
WAV 四要素换算（`demo2.py` 负责这一段，状态机不感知）：

| 采集参数 | 本文依赖 | 取错的后果 |
|----------|----------|-----------|
| `sample_rate`（设备常见 44.1k/48k） | `bytes_per_ms`、`block_samples`、`ring_capacity_ms` | 时间轴整体缩放，`min_silence=300ms` 实际变成 200 或 440ms，切点系统性漂移 |
| `channels`（单麦 1、双麦 2、阵列 4+） | 强制 `== 1`（`validate` 直接拒绝） | 交织数据当单声道读 ⇒ 时长算错一半、切片跨样本、出现爆音 |
| 位宽（int16 LE） | `rms_db` 以 32768 为满量程 | 电平/底噪估计全错，`dispatch_min_level_db` 门限失准，段被误判 `low_level` 丢弃 |
| 设备回调块（`--blocksize`，默认 10ms） | 必须能凑满 `block_ms=32` 的帧栅格 | 帧边界不对齐则派发边界（§5.5 三条约定之二）失效 |

`demo2.py` 的三个设计点与本文的对应关系：

* **多通道收敛**：`--mix mean|energy|copy` 先把多路落成一路；`--mix multi` 则是
  **每通道一套独立状态机 + 独立环形缓冲**（双麦各自分段、各自送 ASR，互不等待），
  此时 `utterance_id` 只在各自的流内单调，跨通道不可比。
* **重采样**：整数比（48k→16k）用箱式平均抽取（等效短窗低通），非整数比
  （44.1k→16k）用跨块连续的线性插值；判决与段音频统一在 16k 时间坐标上。
* **丢弃与空洞（G12）**：采集侧丢块时**只把音频时钟推前、不喂数据**，
  于是 `gap_force_close_ms=200` 会在真的丢帧时强制闭合，段上带 `status=gap`；
  这与"把丢掉的音频当作没发生"相比，代价是可能多切一句，收益是不会把 0.5s 的
  缺失音频混进段里静默丢字。真机若频繁出现该日志，说明消费端跟不上（`--qsize`
  或 VAD 算力问题），应看 `stats.report()["gap_ms"]` 而不是调大 `gap_force_close_ms`。

> 现状缺口：E7 强制闭合目前复用 `CloseReason.EOS`（`engine.py:444`），下游只能靠
> `status == gap` 区分"采集丢帧截断"与"会话结束"。若 ASR 侧需要据此决定是否拼接，
> 应新增 `CloseReason.GAP`，同步 §5.3 转移表与 TC16。

### 9.1 与现有原型 `com/robot/models/asr/asr.py` 的常量对照

`asr.py` 里 `SenseVoice + FSMN-VAD` 的实时链路已经在跑一套**事实上的**分段逻辑，
本文的状态机是它的显式化，不是另起炉灶。对照关系（改任何一边都要同步另一边）：

| `asr.py` | 值 | 本文 | 值 | 差别 / 为什么改 |
|----------|----|------|----|------------------|
| `BUFFER_SECONDS` | 6 s | `ring_capacity_ms` | 8192 ms | 6 s 在 `max_utterance=5000` 下不满足 §4 不等式（G5），6 s 会随机丢字 |
| `VAD_STEP_MS` | 500 | `WindowSegmentVad.stride_ms` | 512 | 必须落在帧栅格上；`stride − block = 480 ms` 就是 `vad_lag` 该覆盖的固有滞后（G19） |
| `VAD_WINDOW_SEC` | 2.5 s（每 0.5 s 重算整个窗） | `window_ms` | 1984 | 冗余计算比 5× → 3.9×；且只交付**新增步长**那批帧概率，不再重复判决同一帧 |
| `SILENCE_PADDING_MS` | 300 | `min_silence_ms` | 300 | 一致。但 FSMN 内部 `max_end_silence_time ≈ 800 ms` 在其**之外**，真实闭合延迟 0.8~1.3 s（G19） |
| `MIN_SPEECH_SEC` | 0.3 | `min_speech_ms` / R6/R7-c | 300 | 一致 |
| `MAX_SPEECH_SEC` | 5.0 | `max_utterance_ms` / R5 | 5000 | 一致；本文补了截断点择优与同句归属（§5.5） |
| `ONGOING_MARGIN_SEC` | 0.25 | `ongoing_margin_ms` | 250 | 一致，但从 `asr.py` 的段列表判读上移到适配器，状态机只看到帧概率 |
| `parse_vad_result` | 手工解析 `value=[[start,end]]`（毫秒） | `parse_segment_result` + `WindowSegmentVad` | 同单位 | 单位是**毫秒**这个坑保留在注释与 TC19 断言里 |
| `NEW_SEG_EPS` | 0.10 s（"防重复识别"） | R7-b `S ≥ D` | 精确 | 原型用 100 ms 容差近似「这段是上次之后新出现的」；本文把它变成显式游标，容差不需要了 |
| `CHUNK_MS` | 100 | `block_ms` | 32 | 100 ms 粒度会让 pre/post_roll（200/100）无法对齐；32 ms = 512 样本 @16k |

`asr.py` 尾注自己写的「每 0.5 s 重算整个 2.5 s 窗口 ≈ 每秒重复计算约 5 倍音频（建议改流式 VAD）」
与 §7-G4/§7-G19 是同一个问题，两处结论一致。

---

## 10. 待确认决策（需要产品/硬件侧回答）

1. 起口/闭合延迟能否接受 ~1 s？若不能，是否启用 §5.7 partial（代价：ASR 重复计算）。
   —— **现状**：默认 batch；`emit_partial=True` 已可用，`Segment.partial` 与同 `seg_id` 覆盖协议已实现（TC13）。
2. 硬件是否有 AEC？决定 G9 走 `half` 还是 `full`（影响能否打断机器人说话）。
   —— **现状**：`gate_mode` 两档都实现了（TC9/TC9c），只等产品确认硬件。
3. 环形缓冲 8 s（G5 方案①）还是上限降到 4.2 s（方案②）？
   —— **现状**：按方案①落库（`ring_capacity_ms=8192`），不等式在 `validate()` 里硬校验，配错启动即 `ConfigError`（TC12）。
4. `min_silence_ms` 是否按 G1 改为 300/650 两级？需要一批真实对话录音做碎片率标定。
   —— **现状**：两级参数都在（300 闭合 / 650 句末二次确认，后者仅 `emit_partial` 生效）；标定后只需改数值。
5. 段是否需保留供 `memory`（人物记忆/大事记）二次消费？若需，`Segment.samples` 的落盘/淘汰策略另议。
6. 唤醒态（G8）与 `emotion` 模块的关系：armed 期间是否受"情绪"影响主动发言阈值？（属跨模块设计，本文不解决。）
7. **VAD 后端选哪个**（新增，实现期发现）：帧概率型（sherpa-onnx silero，需 `.onnx` 模型 + RKNN 适配）
   还是段输出型（FunASR FSMN，仓库现有链路在用）。二者都能接 `ISegmentVAD`，但延迟口径不同：
   FSMN 的 `≈800 ms` 内部尾静音叠加在状态机之外（§7-G19），silero 没有这一层。
   若继续用 FSMN，`asr.py` 里「建议改流式 VAD」这条 TODO 就是本模块的下一步；若换 silero，
   需要确认 RK35xx 上的推理耗时预算（readme 记的是 SenseVoice 已占 CPU/NN 口）。
