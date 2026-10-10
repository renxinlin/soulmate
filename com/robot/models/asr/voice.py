import sounddevice as sd
import numpy as np
import wave

duration = 3  # 录音3秒
fs = 16000
print("🎤 开始录音，请说话...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device='default')
sd.wait()
print("✅ 录音结束")
print(f"最大振幅: {np.max(np.abs(recording))}")

# 可选：保存为 WAV 文件，方便播放检查
with wave.open('test.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes((recording * 32767).astype(np.int16).tobytes())
print("已保存 test.wav，请用播放器检查是否有声音")