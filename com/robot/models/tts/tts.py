
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import subprocess

import unidic_lite
import sounddevice as sd
import soundfile as sf
# 将 unidic 的词典路径强制指向 unidic-lite 的路径

from melo.api import TTS
# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "我的名字叫任子安，我准备上南理工幼儿园。"
model = TTS(language='ZH', device=device)
# model = TTS(language='ZH', device=device, use_hf=False ,config_path = 'path/config.json', ckpt_path='path/checkpoint.pth')
speaker_ids = model.hps.data.spk2id
#
# # audio = model.tts_to_file(text, speaker_ids['ZH'], None, speed=speed)
output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)
#
# #
# # # 采样率
# sr = model.hps.data.sampling_rate
# #
# # # 直接播放
# # sd.play(audio, sr) # wsl环境这里查找不到音频播放设备，但是这rk开发版这里可以查到，使用下方代码模拟播放
# # sd.wait()  # 等待播放结束
#
# """
# 1 支持配置变量，将其中的配置抽出来
# 2 支持多种模型配置，当前要支持meloTTS低端设备，Qwen3-tts高端设备
# 3 支持语言，暂定中英日韩等主流语言
# 4 支持mock Windows wsl ,unbutu ,rk系列开发版以及仿真播放
# """
#
# #
# # # 1. 生成 WAV 文件到 WSL 可访问的 Windows 路径（例如 /mnt/c/...）
# output_path_wsl = '/mnt/c/Users/tyt/PyCharmMiscProject/com/robot/models/tts/zh.wav'
# model.tts_to_file(text, speaker_ids['ZH'], output_path_wsl, speed=speed)
# #
# # # 2. 将 WSL 路径转换为 Windows 路径
# windows_path = subprocess.check_output(['wslpath', '-w', output_path_wsl]).decode().strip()
# # # 例如得到 C:\Users\tyt\...\zh.wav
# #
# # # 3. 使用 PowerShell 播放（阻塞直到播放完成）
# subprocess.run(['powershell.exe', '-c', f'(New-Object Media.SoundPlayer "{windows_path}").PlaySync()'])


model.export_onnx(1)