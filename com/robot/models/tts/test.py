# diagnose.py
import os
import subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np, torch, soundfile
from melo.api import TTS
from melo import utils, commons
from onnx_infer2 import OnnxTTS as TTS_Onnx

TEXT = "今天天气很不错。"     # 先跑中文；再换成 "Today is a nice day." 跑一次英文
LANG = 'ZH_MIX_EN'

tts  = TTS(language='ZH', device='cpu')
otts = TTS_Onnx(language='ZH', encoder_path='encoder-ZH_MIX_EN.onnx', decoder_path='decoder-ZH_MIX_EN.onnx'
                  ,hps = tts.hps,symbol_to_id=tts.symbol_to_id
    , device='cpu')
hps, dev, s2i = tts.hps, tts.device, tts.symbol_to_id
SID, SR = hps.data.spk2id['ZH'], hps.data.sampling_rate

bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(TEXT, LANG, hps, dev, s2i)
n = phones.size(0)

# ① torch 完整 infer（基准）
with torch.no_grad():
    a = tts.model.infer(phones.unsqueeze(0).to(dev), torch.LongTensor([n]).to(dev),
        torch.LongTensor([SID]).to(dev), tones.unsqueeze(0).to(dev),
        lang_ids.unsqueeze(0).to(dev), bert.unsqueeze(0).to(dev),
        ja_bert.unsqueeze(0).to(dev),
        sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, length_scale=1.0)[0][0,0].cpu().numpy()
soundfile.write('diag1_torch_infer.wav', a, SR)

# ② torch 走“导出路径”（forward_encoder/decoder，真实输入、无 padding）
with torch.no_grad():
    logw, x_mask, g, m_p, logs_p = tts.model.forward_encoder(
        phones.unsqueeze(0).to(dev), torch.LongTensor([n]).to(dev),
        torch.LongTensor([SID]).to(dev), tones.unsqueeze(0).to(dev),
        lang_ids.unsqueeze(0).to(dev), ja_bert.unsqueeze(0).to(dev),   # 只喂 ja_bert，与导出一致
        sdp_ratio=0.2, noise_scale_w=0.8)
    w_ceil = torch.ceil(torch.exp(logw) * x_mask)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1,2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn = commons.generate_path(w_ceil, torch.unsqueeze(x_mask,2)*torch.unsqueeze(y_mask,-1)).squeeze(1)
    y = tts.model.forward_decoder(attn, y_mask, g, m_p, logs_p, 0.6)
    if isinstance(y, (tuple, list)): y = y[0]
soundfile.write('diag2_torch_exportpath.wav', y.reshape(-1).cpu().numpy(), SR)

# ③ onnx（带 padding，与真实推理一致）中间量对比
mx = 256
rp = lambda a: np.pad(a, ((0,0),(0, mx-a.shape[1]))) if a.shape[1] < mx else a
feed = {'x': rp(phones.unsqueeze(0).cpu().numpy()),
        'x_lengths': np.array([n], np.int64), 'sid': np.array([SID], np.int64),
        'tone': rp(tones.unsqueeze(0).cpu().numpy()),
        'lang_ids': rp(lang_ids.unsqueeze(0).cpu().numpy()),
        'ja_bert': np.pad(ja_bert.unsqueeze(0).cpu().numpy(), ((0,0),(0,0),(0,mx-n))),
        'noise_scale_w': np.array([0.8], np.float32),
        'sdp_ratio': np.array([0.2], np.float32)}
_, _, g_o, mp_o, _ = otts.encoder_sess.run(['logw','x_mask','g','m_p','logs_p'], feed)
print('decoder 的 g 形状 =', g_o.shape,
      '→ [1,C,1]=时间池化(受补零影响)；[1,C,T]=逐帧(不受影响)')
print(f'g   max|Δ| = {np.abs(g_o - g.cpu().numpy()).max():.4f}    (大 → 均值被稀释，原因3)')
print(f'm_p max|Δ| = {np.abs(mp_o[:,:,:n] - m_p.cpu().numpy()[:,:,:n]).max():.4f}    (大 → 图丢了中文 BERT，原因2)')