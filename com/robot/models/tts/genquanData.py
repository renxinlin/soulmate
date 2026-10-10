# -*- coding: utf-8 -*-
"""生成 int8 量化校准数据集 + 仿真对比用 golden 样本"""
import os, json, re
import numpy as np
import onnxruntime as ort
from melo import utils
from melo.split_utils import split_sentence

X_MAX_LEN = 256
OUT_DIR = 'calib'
CALIB_TEXTS = [
    "我的名字叫任子安, 我准备上南理工幼儿园.",
    "我最近在学习machine learning。",
    "今天天气怎么样？我们一起出去玩吧。",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "Hello, this is a test sentence for calibration.",
]
RNG = np.random.RandomState(1234)

class HParams:
    def __init__(self, **kw):
        for k, v in kw.items():
            if type(v) == dict: v = HParams(**v)
            setattr(self, k, v)
    def __getitem__(self, key): return getattr(self, key)

def sequence_mask(lengths, max_length=None):
    lengths = np.asarray(lengths)
    if max_length is None: max_length = int(lengths.max())
    x = np.arange(max_length)
    return (x[None, :] < lengths[:, None]).astype(np.float32)

def generate_path(duration, mask):
    b, _, t_y, t_x = mask.shape
    cum = np.cumsum(duration, -1)
    path = sequence_mask(cum.reshape(b * t_x), t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    path = path - np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    return path[:, None].transpose(0, 1, 3, 2) * mask

def load_manifest(path):
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    return None

def build_manifest(feed, default_order):
    manifest = []
    for name in default_order:
        if name in feed:
            a = feed[name]
            manifest.append({'name': name, 'shape': list(a.shape),
                             'dtype': 'INT32' if np.issubdtype(a.dtype, np.integer) else 'FLOAT32'})
    return manifest

def cast_by_meta(a, meta):
    return np.ascontiguousarray(
        np.asarray(a).astype(np.int32 if meta.get('dtype') == 'INT32' else np.float32))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hps = HParams(**json.load(open('config.json', encoding='utf-8')))
    symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}

    enc_manifest = load_manifest('encoder-ZH_MIX_EN.fixed.onnx.inputs.json') or None
    dec_manifest = load_manifest('decoder-ZH_MIX_EN.fixed.onnx.inputs.json') or None

    enc_sess = ort.InferenceSession('encoder-ZH_MIX_EN.onnx', providers=['CPUExecutionProvider'])
    dec_sess = ort.InferenceSession('decoder-ZH_MIX_EN.onnx', providers=['CPUExecutionProvider'])

    enc_lines, dec_lines = [], []
    golden_enc, golden_dec, golden_dec_in = None, None, None

    for idx, text in enumerate(CALIB_TEXTS):
        for t in split_sentence(text, language_str='ZH_MIX_EN'):
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            _, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                t, 'ZH_MIX_EN', hps, 'cpu', symbol_to_id)
            pad2 = lambda a: np.concatenate([a, np.repeat(a[:, -1:], X_MAX_LEN - a.shape[1], 1)], 1) \
                if a.shape[1] < X_MAX_LEN else a
            x = pad2(phones.numpy().reshape(1, -1))
            tone = pad2(tones.numpy().reshape(1, -1))
            lang = pad2(lang_ids.numpy().reshape(1, -1))
            jb = ja_bert.numpy().astype(np.float32)
            jb = jb[np.newaxis] if jb.ndim == 2 else jb
            jb = jb.transpose(0, 2, 1) if jb.shape[1] == 768 else jb
            jb = np.concatenate([jb, np.repeat(jb[:, :, -1:], X_MAX_LEN - jb.shape[2], 2)], 2)

            enc_feed = {'x': x, 'x_lengths': np.array([phones.size(0)]),
                        'sid': np.array([1]).reshape(1), 'tone': tone, 'lang_ids': lang,
                        'ja_bert': jb, 'noise_scale_w': np.array([0.8], np.float32),
                        'sdp_ratio': np.array([0.2], np.float32)}
            if enc_manifest is None:
                enc_manifest = build_manifest(
                    enc_feed, ['x', 'x_lengths', 'sid', 'tone', 'lang_ids',
                               'ja_bert', 'noise_scale_w', 'sdp_ratio'])

            paths = []
            for meta in enc_manifest:
                if meta['name'] in enc_feed:
                    a = enc_feed[meta['name']]
                elif meta.get('noise'):
                    a = RNG.randn(*meta['shape']).astype(np.float32)
                else:
                    raise KeyError(meta['name'])
                p = f'{OUT_DIR}/enc_{idx:02d}_{meta["name"]}.npy'
                np.save(p, cast_by_meta(a, meta))
                paths.append(p)
            enc_lines.append(' '.join(paths))
            if golden_enc is None:
                golden_enc = {m['name']: cast_by_meta(
                    enc_feed.get(m['name'], RNG.randn(*m['shape'])), m) for m in enc_manifest}

            logw, x_mask, g, m_p, logs_p = enc_sess.run(None, {
                'x': x.astype(np.int64), 'x_lengths': np.array([phones.size(0)], np.int64),
                'sid': np.array([1], np.int64), 'tone': tone.astype(np.int64),
                'lang_ids': lang.astype(np.int64), 'ja_bert': jb,
                'noise_scale_w': np.array([0.8], np.float32),
                'sdp_ratio': np.array([0.2], np.float32)})

            w_ceil = np.ceil(np.exp(logw) * x_mask)
            y_mask = np.ones((1, 1, 2 * X_MAX_LEN), np.float32)
            attn = generate_path(w_ceil, x_mask[:, :, None, :] * y_mask[:, :, :, None]).squeeze(1)

            dec_feed = {'attn': attn.astype(np.float32), 'y_mask': y_mask,
                        'g': g, 'm_p': m_p, 'logs_p': logs_p,
                        'noise_scale': np.array([0.6], np.float32)}
            if dec_manifest is None:
                dec_manifest = build_manifest(
                    dec_feed, ['attn', 'y_mask', 'g', 'm_p', 'logs_p', 'noise_scale'])

            paths = []
            for meta in dec_manifest:
                if meta['name'] in dec_feed:
                    a = dec_feed[meta['name']]
                elif meta.get('noise'):
                    a = RNG.randn(*meta['shape']).astype(np.float32)
                else:
                    raise KeyError(meta['name'])
                p = f'{OUT_DIR}/dec_{idx:02d}_{meta["name"]}.npy'
                np.save(p, cast_by_meta(a, meta))
                paths.append(p)
            dec_lines.append(' '.join(paths))
            if golden_dec_in is None:
                golden_dec_in = {m['name']: cast_by_meta(
                    dec_feed.get(m['name'], RNG.randn(*m['shape'])), m) for m in dec_manifest}

    open(f'{OUT_DIR}/dataset_encoder.txt', 'w').write('\n'.join(enc_lines) + '\n')
    open(f'{OUT_DIR}/dataset_decoder.txt', 'w').write('\n'.join(dec_lines) + '\n')
    json.dump(enc_manifest, open(f'{OUT_DIR}/manifest_enc.json', 'w'))
    json.dump(dec_manifest, open(f'{OUT_DIR}/manifest_dec.json', 'w'))
    np.savez(f'{OUT_DIR}/golden_enc.npz', **golden_enc)
    np.savez(f'{OUT_DIR}/golden_dec.npz', **golden_dec_in)
    print(f'校准集: {len(enc_lines)} 组 (encoder/decoder 各一份)')

if __name__ == '__main__':
    main()
