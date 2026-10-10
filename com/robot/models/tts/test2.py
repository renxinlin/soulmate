# diagnose.py —— 独立诊断: 定位"机器人声"是丢中文BERT还是g被补零稀释
import os
import subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np, torch, soundfile, onnxruntime as ort
from melo.api import TTS
from melo import utils, commons

TEXT = "今天天气很不错。"   # 第一轮用中文句(测 m_p)；第二轮换 "Today is a nice day."(测 g)
LANG, ENC_PATH, DEC_PATH = 'ZH_MIX_EN', 'encoder-ZH_MIX_EN.onnx', 'decoder-ZH_MIX_EN.onnx'

tts = TTS(language='ZH', device='cpu')
hps, dev, s2i = tts.hps, tts.device, tts.symbol_to_id
SID, SR = hps.data.spk2id['ZH'], hps.data.sampling_rate

enc = ort.InferenceSession(ENC_PATH, providers=['CPUExecutionProvider'])
dec = ort.InferenceSession(DEC_PATH, providers=['CPUExecutionProvider'])
print('encoder inputs :', [(i.name, i.shape) for i in enc.get_inputs()])
print('decoder inputs :', [(i.name, i.shape) for i in dec.get_inputs()])

bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(TEXT, LANG, hps, dev, s2i)
n = phones.size(0)

# ① torch 完整 infer(金标准)
with torch.no_grad():
    a = tts.model.infer(phones.unsqueeze(0).to(dev), torch.LongTensor([n]).to(dev),
        torch.LongTensor([SID]).to(dev), tones.unsqueeze(0).to(dev), lang_ids.unsqueeze(0).to(dev),
        bert.unsqueeze(0).to(dev), ja_bert.unsqueeze(0).to(dev),
        sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, length_scale=1.0)[0][0, 0].cpu().numpy()
soundfile.write('diag1_torch_infer.wav', a, SR)

# ② torch 走导出路径(只喂 ja_bert, 无 padding)
with torch.no_grad():
    logw, x_mask, g, m_p, logs_p = tts.model.forward_encoder(
        phones.unsqueeze(0).to(dev), torch.LongTensor([n]).to(dev),
        torch.LongTensor([SID]).to(dev), tones.unsqueeze(0).to(dev),
        lang_ids.unsqueeze(0).to(dev), ja_bert.unsqueeze(0).to(dev),
        sdp_ratio=0.2, noise_scale_w=0.8)
    w_ceil = torch.ceil(torch.exp(logw) * x_mask)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn = commons.generate_path(w_ceil, torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)).squeeze(1)
    y = tts.model.forward_decoder(attn, y_mask, g, m_p, logs_p, 0.6)
    if isinstance(y, (tuple, list)): y = y[0]
soundfile.write('diag2_torch_exportpath.wav', y.reshape(-1).cpu().numpy(), SR)

# ③ onnx(带 padding, 与真实推理一致) + 中间量对比
def sdim(shape, idx):
    try: d = shape[idx]
    except Exception: return None
    return d if isinstance(d, int) and d > 0 else None

enc_in = {i.name: i for i in enc.get_inputs()}
mx = sdim(enc_in['x'].shape, 1); pad = (mx - n) if (mx and n < mx) else 0
rp = lambda a, ax: np.pad(a, [(0, 0)] * ax + [(0, pad)]) if pad else a
feed = {'x': rp(phones.unsqueeze(0).cpu().numpy().astype(np.int64), 1),
        'x_lengths': np.array([n], np.int64), 'sid': np.array([SID], np.int64),
        'tone': rp(tones.unsqueeze(0).cpu().numpy().astype(np.int64), 1),
        'lang_ids': rp(lang_ids.unsqueeze(0).cpu().numpy().astype(np.int64), 1),
        'ja_bert': rp(ja_bert.unsqueeze(0).cpu().numpy().astype(np.float32), 2)}
if 'bert' in enc_in:    feed['bert'] = rp(bert.unsqueeze(0).cpu().numpy().astype(np.float32), 2)
if 'noise_scale_w' in enc_in: feed['noise_scale_w'] = np.array([0.8], np.float32)
if 'sdp_ratio' in enc_in:     feed['sdp_ratio']     = np.array([0.2], np.float32)

logw_o, x_mask_o, g_o, m_p_o, logs_p_o = enc.run(['logw', 'x_mask', 'g', 'm_p', 'logs_p'], feed)
print(f"g   max|Δ| = {np.abs(g_o - g.cpu().numpy()).max():.4f}   (英文句大 → g 均值被补零稀释)")
print(f"m_p max|Δ| = {np.abs(m_p_o[:, :, :n] - m_p.cpu().numpy()).max():.4f}   (中文句大 → 图里丢了 1024 维中文 BERT)")

logw_t = torch.from_numpy(np.ascontiguousarray(logw_o)); xm = torch.from_numpy(np.ascontiguousarray(x_mask_o))
w_ceil = torch.ceil(torch.exp(logw_t) * xm)
y_len = int(torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()[0])
ym = torch.unsqueeze(commons.sequence_mask(torch.LongTensor([y_len]), None), 1).to(xm.dtype)
at = commons.generate_path(w_ceil, torch.unsqueeze(xm, 2) * torch.unsqueeze(ym, -1)).squeeze(1)
if at.dim() == 2: at = at.unsqueeze(1)

def fit(arr, meta):
    ps = [(0, t - c) if isinstance(t, int) and t > 0 and c < t else (0, 0)
          for c, t in zip(arr.shape, meta.shape)]
    return np.pad(arr, ps) if any(p[1] for p in ps) else arr

dm = {i.name: i for i in dec.get_inputs()}
dec_in = {'attn': np.ascontiguousarray(at.numpy(), np.float32),
          'y_mask': np.ascontiguousarray(ym.numpy(), np.float32),
          'g': g_o, 'm_p': m_p_o, 'logs_p': logs_p_o,
          'noise_scale': np.array([0.6], np.float32)}
for k in ('attn', 'y_mask', 'm_p', 'logs_p'):
    if k in dm: dec_in[k] = fit(dec_in[k], dm[k])

y_o = dec.run(['y'], dec_in)[0]
nsamp = min(y_o.shape[-1], y_len * hps.data.hop_length)
soundfile.write('diag3_onnx.wav', y_o[0, 0, :nsamp].astype(np.float32), SR)
print('已生成 diag1_torch_infer / diag2_torch_exportpath / diag3_onnx 三个 wav，试听对比')

print('x_mask 前 n 位 / 后面:', x_mask_o[0, 0, :n].min(), x_mask_o[0, 0, n:].max())  # 期望 1.0 / 0.0
feed2 = dict(feed); feed2['x_lengths'] = np.array([256], np.int64)
_, _, _, mp2, _ = enc.run(['logw','x_mask','g','m_p','logs_p'], feed2)
print('改 x_lengths 有影响吗:', np.abs(mp2[:, :, :n] - m_p_o[:, :, :n]).max())       # >0 才正常

