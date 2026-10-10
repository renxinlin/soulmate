import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import time
import threading
import numpy as np
import sounddevice as sd
from funasr import AutoModel

# ==================== 1. 配置参数 ====================
SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2          # int16 单声道 = 2字节/样本
CHUNK_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
BUFFER_SECONDS = 6
BUFFER_SIZE = BYTES_PER_SEC * BUFFER_SECONDS
VAD_STEP_MS = 500
VAD_STEP_SIZE = int(SAMPLE_RATE * VAD_STEP_MS / 1000) * 2
SILENCE_PADDING_MS = 300
MIN_SPEECH_SEC = 0.3
MAX_SPEECH_SEC = 5.0

VAD_WINDOW_SEC = 2.5                     # 每次送VAD的最近音频长度
ONGOING_MARGIN_SEC = 0.25                # 段尾距窗口末端 < 该值 => 仍在说话
NEW_SEG_EPS = int(0.10 * BYTES_PER_SEC)  # 容忍的时间戳抖动(防重复识别)
DEBUG = True

# ==================== 2. 初始化模型 ====================
print("正在加载模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cpu",
    disable_update=True,
    # disable_auto_download=True,
)
vad_model = AutoModel(
    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    device="cpu",
    disable_update=True,
    # disable_auto_download=True,
)
print("模型加载完成！")

# ==================== 3. 环形缓冲区（原样） ====================
class RingBuffer:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.buffer = bytearray(max_bytes)
        self.write_pos = 0
        self.total_written = 0

    def append(self, data: bytes):
        data_len = len(data)
        if data_len >= self.max_bytes:
            self.buffer = bytearray(data[-self.max_bytes:])
            self.write_pos = 0
            self.total_written += data_len
            return
        end_pos = self.write_pos + data_len
        if end_pos <= self.max_bytes:
            self.buffer[self.write_pos:end_pos] = data
        else:
            first_part = self.max_bytes - self.write_pos
            self.buffer[self.write_pos:] = data[:first_part]
            self.buffer[:data_len - first_part] = data[first_part:]
        self.write_pos = end_pos % self.max_bytes
        self.total_written += data_len

    def get_audio(self, start_byte_offset, length_bytes):
        if length_bytes > self.max_bytes:
            raise ValueError("请求长度超过缓冲区大小")
        start = start_byte_offset % self.max_bytes
        end = start + length_bytes
        if end <= self.max_bytes:
            return bytes(self.buffer[start:end])
        part1 = self.buffer[start:]
        part2 = self.buffer[:end - self.max_bytes]
        return bytes(part1 + part2)

# ==================== 4. 音频采集线程 ====================
ring = RingBuffer(BUFFER_SIZE)
audio_running = True

def audio_capture():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK_SIZE, dtype='float32') as stream:
        while audio_running:
            data, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("录音溢出")
            data = np.clip(data, -1.0, 1.0)      # 防止越界导致int16回绕
            ring.append((data * 32767).astype(np.int16).tobytes())

threading.Thread(target=audio_capture, daemon=True).start()
print("开始实时语音识别，按 Ctrl+C 停止...")

# ==================== 5. 工具函数 ====================
def parse_vad_result(vad_res):
    """
    ★★★ 核心修复: FSMN-VAD 的 value=[[start,end]] 单位是【毫秒】不是采样点!
    原代码除以 sample_rate, 导致所有段长 <= 0.155s, 状态机永远不启动。
    返回 [(start_sec, end_sec), ...]
    """
    segments = []
    if not vad_res:
        return segments
    for item in vad_res:
        if not isinstance(item, dict):
            continue
        value = item.get('value')
        if not value or not isinstance(value, list):
            continue
        for seg in value:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                s_ms, e_ms = seg
                if e_ms > s_ms:
                    segments.append((s_ms / 1000.0, e_ms / 1000.0))  # ms -> 秒
    return segments

TAG_RE = re.compile(r'<\|[^|>]*\|>')    # 去掉 <|zh|><|NEUTRAL|><|Speech|> 等标记

def recognize(pcm_bytes):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio.size == 0:
        return
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    res = model.generate(input=audio, language="auto", use_itn=True)
    if res and len(res) > 0:
        text = TAG_RE.sub('', res[0].get('text', '')).strip()
        if text:
            print(f"👉 {text}")

# ==================== 6. 主循环：VAD状态机 ====================
speech_start_byte = None
speech_end_byte = None        # 最近一次检测到语音的末端
is_speaking = False
last_vad_pos = 0
processed_end_byte = 0        # ★ 已处理语音的末端, 跳过窗口里的残留, 防止重复识别

try:
    while True:
        current_total = ring.total_written

        if last_vad_pos == 0:
            last_vad_pos = current_total
            time.sleep(0.2)
            continue
        if current_total - last_vad_pos < VAD_STEP_SIZE:
            time.sleep(0.05)
            continue

        # ---- 取最近2.5秒【真实】音频(启动期不足则按实际长度, 不补零) ----
        full_len = int(VAD_WINDOW_SEC * BYTES_PER_SEC)
        start_byte = max(0, current_total - full_len)
        analyze_len = current_total - start_byte          # 窗口末端 == 当前时刻
        audio_np = np.frombuffer(ring.get_audio(start_byte, analyze_len), dtype=np.int16)
        window_sec = analyze_len / BYTES_PER_SEC

        vad_res = vad_model.generate(input=audio_np, cache={}, fs=SAMPLE_RATE)
        segments = parse_vad_result(vad_res)
        if DEBUG:
            print(f"音频均值: {np.mean(np.abs(audio_np)):.1f}  segments: {segments}")

        # ★ 只有语音"延伸到窗口末端"才算还在说话, 窗口前部的旧语音不算
        speech_ongoing = any(
            rel_end >= window_sec - ONGOING_MARGIN_SEC for _, rel_end in segments
        )

        # ---------- 空闲: 寻找"未处理过"的新语音 ----------
        if not is_speaking:
            for rel_start, rel_end in segments:
                abs_start = start_byte + int(rel_start * BYTES_PER_SEC)
                abs_end   = start_byte + int(rel_end * BYTES_PER_SEC)
                if abs_end <= processed_end_byte + NEW_SEG_EPS:   # 上一句的残留
                    continue
                if not speech_ongoing and (rel_end - rel_start) < MIN_SPEECH_SEC:
                    continue                                       # 已结束且太短
                speech_start_byte = max(abs_start, processed_end_byte)
                speech_end_byte = abs_end
                is_speaking = True
                print("🎤 检测到语音开始...")
                break

        # ---------- 说话中: 跟踪末端 / 判断结束 ----------
        if is_speaking:
            for _, rel_end in segments:
                abs_end = start_byte + int(rel_end * BYTES_PER_SEC)
                if abs_end > speech_end_byte:
                    speech_end_byte = abs_end

            # ★ 静音时长直接 = 当前时刻 - 最近语音末端(不再用0.5s步进累加)
            silence_sec = 0.0 if speech_ongoing else \
                max(0.0, (current_total - speech_end_byte) / BYTES_PER_SEC)

            # ---- 超长强制截断 ----
            if (current_total - speech_start_byte) > MAX_SPEECH_SEC * BYTES_PER_SEC:
                print(f"⚠️ 语音超过{MAX_SPEECH_SEC}秒，强制截断")
                speech_start_byte = max(speech_start_byte,
                                        current_total - BUFFER_SIZE + int(0.2 * BYTES_PER_SEC))
                voice_len = min(int(MAX_SPEECH_SEC * BYTES_PER_SEC),
                                current_total - speech_start_byte)
                if voice_len > MIN_SPEECH_SEC * BYTES_PER_SEC:
                    recognize(ring.get_audio(speech_start_byte, voice_len))
                processed_end_byte = speech_start_byte + voice_len
                is_speaking = False
                speech_start_byte = speech_end_byte = None
                last_vad_pos = current_total
                continue

            # ---- 静音超时 => 语音结束 ----
            if silence_sec >= SILENCE_PADDING_MS / 1000.0:
                # ★ 按"真实语音末端 + 300ms"截取(原代码的padding是no-op)
                end_byte = min(speech_end_byte + int(SILENCE_PADDING_MS / 1000 * BYTES_PER_SEC),
                               current_total)
                speech_start_byte = max(speech_start_byte,
                                        current_total - BUFFER_SIZE + int(0.2 * BYTES_PER_SEC))
                voice_len = max(0, end_byte - speech_start_byte)
                if voice_len > MIN_SPEECH_SEC * BYTES_PER_SEC:
                    recognize(ring.get_audio(speech_start_byte, voice_len))
                else:
                    print("⏳ 语音太短，已忽略")
                processed_end_byte = end_byte
                is_speaking = False
                speech_start_byte = speech_end_byte = None

        last_vad_pos = current_total
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n停止检测...")
    audio_running = False

"""
segments 会打印出秒级的合理值，如 [(2.06, 2.48)] 表示窗口尾部 0.42s 有语音且仍在说话；说完后约 0.8~1.3 秒出文字。这个延迟下限由 FSMN-VAD 自身的 max_end_silence_time（默认约 800ms 静音才确认"说完了"）+ 500ms 检测步长决定。
如果发现句中被截断，把 SILENCE_PADDING_MS 调大到 500~800；嫌结束慢就调小。
你日志开头在近静音（均值 0.87）时 VAD 也报了语音，说明你的麦克风底噪不低——必要时可在送 VAD 前加一个能量门限。
更标准的做法是用 FunASR 的流式 VAD（vad_model.generate(input=chunk, cache={}, chunk_size=200, is_final=False)），增量消费音频；目前"每 0.5s 重算整个 2.5s 窗口"相当于每秒重复计算约 5 倍的音频，CPU 上虽然跑得动，但流式方案更干净、延迟也更低。"""