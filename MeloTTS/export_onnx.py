import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
# text = "Did you ever hear a folk tale about a giant turtle?"

model = TTS(language='ZH', device=device, use_hf=False ,config_path = '../../models/MeloTTS-Chinese/config.json', 
            ckpt_path='../../models/MeloTTS-Chinese/checkpoint.pth')
speaker_ids = model.hps.data.spk2id

# output_path = 'zh.wav'
# model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)

model.export_onnx(speaker_ids['ZH'])