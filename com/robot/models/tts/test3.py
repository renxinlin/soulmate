# exp_b.py —— 让 torch 吃和 onnx 完全相同的补零输入
# diagnose.py —— 独立诊断: 定位"机器人声"是丢中文BERT还是g被补零稀释
import os
import subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np, torch, torch.nn.functional as F
import onnxruntime as ort
from melo.api import TTS
from melo import utils

TEXT, LANG = "今天天气很不错。", "ZH_MIX_EN"
tts = TTS(language='ZH', device='cpu')
hps, s2i = tts.hps, tts.symbol_to_id
SID = hps.data.spk2id['ZH']

bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(TEXT, LANG, hps, 'cpu', s2i)
n = phones.size(0)

enc = ort.InferenceSession('encoder-ZH_MIX_EN.onnx', providers=['CPUExecutionProvider'])
print('enc inputs:', [(i.name, i.shape) for i in enc.get_inputs()])
xmeta = [i for i in enc.get_inputs() if i.name == 'x'][0]
mx = xmeta.shape[1]
assert isinstance(mx, int) and n < mx, f'x.shape={xmeta.shape}'
p = mx - n

feed = {'x': F.pad(phones, (0, p)).unsqueeze(0).numpy(),
        'x_lengths': np.array([n], np.int64), 'sid': np.array([SID], np.int64),
        'tone': F.pad(tones, (0, p)).unsqueeze(0).numpy(),
        'lang_ids': F.pad(lang_ids, (0, p)).unsqueeze(0).numpy(),
        'ja_bert': F.pad(ja_bert.float(), (0, p)).unsqueeze(0).numpy(),
        'noise_scale_w': np.array([0.8], np.float32), 'sdp_ratio': np.array([0.2], np.float32)}
_, _, _, m_p_onnx, _ = enc.run(['logw', 'x_mask', 'g', 'm_p', 'logs_p'], feed)

with torch.no_grad():
    kw = dict(sdp_ratio=0.2, noise_scale_w=0.8)
    _, _, _, m_p_pad, _ = tts.model.forward_encoder(          # torch + 补零
        F.pad(phones, (0, p)).unsqueeze(0), torch.LongTensor([n]), torch.LongTensor([SID]),
        F.pad(tones, (0, p)).unsqueeze(0), F.pad(lang_ids, (0, p)).unsqueeze(0),
        F.pad(ja_bert.float(), (0, p)).unsqueeze(0), **kw)
    _, _, _, m_p_raw, _ = tts.model.forward_encoder(          # torch 不补零
        phones.unsqueeze(0), torch.LongTensor([n]), torch.LongTensor([SID]),
        tones.unsqueeze(0), lang_ids.unsqueeze(0), ja_bert.unsqueeze(0), **kw)
    m = 32                                                     # 附带: 补到小长度再测一次
    _, _, _, m_p_pad32, _ = tts.model.forward_encoder(
        F.pad(phones, (0, m - n)).unsqueeze(0), torch.LongTensor([n]), torch.LongTensor([SID]),
        F.pad(tones, (0, m - n)).unsqueeze(0), F.pad(lang_ids, (0, m - n)).unsqueeze(0),
        F.pad(ja_bert.float(), (0, m - n)).unsqueeze(0), **kw)

a, b, c = m_p_pad[:, :, :n].numpy(), m_p_raw.numpy(), m_p_onnx[:, :, :n]
print(f"m_p std = {b.std():.3f}")
print(f"d1 = |torch补零 - torch不补零| = {np.abs(a - b).max():.4f}  ← 大 ⇒ P1: 补零本身改变模型输出")
print(f"d2 = |torch补零 - onnx|       = {np.abs(a - c).max():.4f}  ← 大 ⇒ P2: onnx 图算错")
print(f"d3 = |torch不补零 - onnx|     = {np.abs(b - c).max():.4f}  (应复现 0.4974)")
print(f"d1@pad32 = {np.abs(m_p_pad32[:, :, :n].numpy() - b).max():.4f}  ← 比 d1 小很多 ⇒ bucket 导出可行")