# -*- coding: utf-8 -*-
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import re, json, subprocess
import numpy as np
import soundfile as sf
from melo import utils
from melo.split_utils import split_sentence

# ============ HParams / 工具: 与你原文件完全一致 ============
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict: v = HParams(**v)
            self[k] = v
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __len__(self): return len(self.__dict__)
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def __contains__(self, key): return key in self.__dict__
    def __repr__(self): return self.__dict__.__repr__()

def sequence_mask(lengths, max_length=None):
    lengths = np.asarray(lengths)
    if max_length is None: max_length = int(lengths.max())
    x = np.arange(max_length)
    return (x[None, :] < lengths[:, None]).astype(np.float32)

def generate_path(duration, mask):
    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, axis=-1)
    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    padded = np.pad(path, ((0, 0), (1, 0), (0, 0)), mode='constant')
    path = path - padded[:, :-1, :]
    path = path[:, None].transpose(0, 1, 3, 2) * mask
    return path

# ============ RKNN session 封装 ============
def make_session(rknn_path, mode="sim", target="rk3566", device_id=None):
    """mode: 'sim'=WSL仿真 / 'adb'=toolkit2经adb连板 / 'lite'=板上RKNNLite"""
    if mode in ("sim", "adb"):
        from rknn.api import RKNN
        s = RKNN(verbose=False)
        assert s.load_rknn(rknn_path) == 0
        if mode == "sim":
            assert s.init_runtime() == 0                      # target=None -> x86仿真
        else:
            assert s.init_runtime(target=target, device_id=device_id) == 0
    else:
        from rknnlite.api import RKNNLite                     # 板上
        s = RKNNLite()
        assert s.load_rknn(rknn_path) == 0
        assert s.init_runtime() == 0
    return s

class RknnTTS:
    def __init__(self, language, encoder_rknn, decoder_rknn,
                 encoder_manifest, decoder_manifest, config_path,
                 mode="sim", target="rk3566", device_id=None):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.hps = HParams(**config)
        self.symbol_to_id = {s: i for i, s in enumerate(self.hps.symbols)}
        self.sampling_rate = self.hps.data.sampling_rate
        self.x_length = 256                     # 与导出时一致(静态shape)

        self.enc_inputs = json.load(open(encoder_manifest, encoding="utf-8"))
        self.dec_inputs = json.load(open(decoder_manifest, encoding="utf-8"))
        self.encoder = make_session(encoder_rknn, mode, target, device_id)
        self.decoder = make_session(decoder_rknn, mode, target, device_id)

    # ---- 按名字喂 dict, 内部转成 RKNN 需要的有序 list ----
    def _run(self, sess, manifest, feed, rng=None):
        ins = []
        for meta in manifest:
            name, shape = meta["name"], meta["shape"]
            if name in feed:
                a = np.asarray(feed[name])
                a = a.astype(np.int32 if meta["dtype"] == "INT32" else np.float32)
            elif meta.get("noise"):            # step1 从 RandomNormal 提出的噪声
                a = (rng or np.random).randn(*shape).astype(np.float32)
            else:
                raise KeyError(f"缺少模型输入: {name}")
            if tuple(shape) and tuple(a.shape) != tuple(shape):
                raise ValueError(f"输入 {name}: got {a.shape}, want {tuple(shape)}")
            ins.append(np.ascontiguousarray(a))
        return sess.inference(inputs=ins)

    def split_sentences_into_pieces(self, text, language):
        texts = split_sentence(text, language_str=language)
        print(" > Text split to sentences.\n" + "\n".join(texts))
        return texts

    def prepare_input(self, text, language, speaker_id):     # 同原文件
        if language in ['EN', 'ZH_MIX_EN']:
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, language, self.hps, 'cpu', self.symbol_to_id)
        x = phones.numpy().reshape(1, -1)
        sid = np.array([speaker_id]).reshape(1)
        tone = tones.numpy().reshape(1, -1)
        lang_ids = lang_ids.numpy().reshape(1, -1)
        x_lengths = np.array([phones.size(0)])
        ja_bert = ja_bert.numpy().astype(np.float32)
        if len(ja_bert.shape) == 2:
            ja_bert = ja_bert[np.newaxis, :, :]
        ja_bert = ja_bert.transpose(0, 2, 1) if ja_bert.shape[1] != 768 else ja_bert
        return x, x_lengths, sid, tone, lang_ids, ja_bert

    def pad_2d(self, arr, target_len):       # 复制末元素填充, 同原文件
        pad_len = target_len - arr.shape[1]
        if pad_len > 0:
            arr = np.concatenate([arr, np.repeat(arr[:, -1:], pad_len, axis=1)], axis=1)
        return arr

    def pad_3d(self, arr, target_len):
        pad_len = target_len - arr.shape[2]
        if pad_len > 0:
            arr = np.concatenate([arr, np.repeat(arr[:, :, -1:], pad_len, axis=2)], axis=2)
        return arr

    def audio_numpy_concat(self, segment_data_list, sr, speed=1.0):  # 同原文件
        res = []
        for seg in segment_data_list:
            res += seg.reshape(-1).tolist()
            res += [0] * int((sr * 0.05) / speed)
        return np.array(res).astype(np.float32)

    def tts(self, text, speaker_id=1, language='ZH_MIX_EN', sdp_ratio=0.2,
            noise_scale=0.6, noise_scale_w=0.8, speed=1.0, rng=None):
        texts = self.split_sentences_into_pieces(text, language)
        audio_list, x_max_len = [], self.x_length
        for t in texts:
            x, x_lengths, sid, tone, lang_ids, ja_bert = \
                self.prepare_input(t, language, speaker_id)
            ja_bert = self.pad_3d(ja_bert, x_max_len)
            lang_ids = self.pad_2d(lang_ids, x_max_len)
            tone = self.pad_2d(tone, x_max_len)
            x = self.pad_2d(x, x_max_len)

            # ================= encoder (RKNN) =================
            enc_feed = {'x': x, 'x_lengths': x_lengths, 'sid': sid, 'tone': tone,
                        'lang_ids': lang_ids, 'ja_bert': ja_bert,
                        'noise_scale_w': np.array([noise_scale_w], np.float32),
                        'sdp_ratio': np.array([sdp_ratio], np.float32)}
            logw, x_mask, g, m_p, logs_p = self._run(
                self.encoder, self.enc_inputs, enc_feed, rng)

            # ================= numpy 中间处理 (同原文件) =================
            length_scale = 1.0 / speed
            w = np.exp(logw) * x_mask * length_scale
            w_ceil = np.ceil(w)
            y_length_max = 2 * x_max_len                        # 512
            y_mask = np.ones((1, 1, y_length_max), np.float32)
            attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
            attn = self.generate_path(w_ceil, attn_mask).squeeze(1)

            # ================= decoder (RKNN) =================
            dec_feed = {'attn': attn.astype(np.float32), 'y_mask': y_mask,
                        'g': g, 'm_p': m_p, 'logs_p': logs_p,
                        'noise_scale': np.array([noise_scale], np.float32)}
            decoder_outputs = self._run(self.decoder, self.dec_inputs, dec_feed, rng)

            audio = decoder_outputs[0][0, 0, :]
            total_frames = int(np.sum(w_ceil * x_mask))
            if 0 < total_frames < audio.shape[0]:
                samples_per_frame = audio.shape[0] // y_mask.shape[2]
                audio = audio[: total_frames * samples_per_frame]
            audio_list.append(audio)
        return self.audio_numpy_concat(audio_list, sr=self.sampling_rate, speed=speed)


if __name__ == '__main__':
    text = "我最近在学习machine learning。"
    tts = RknnTTS(language='ZH',
                  encoder_rknn='encoderk3566.rknn',
                  decoder_rknn='decoder3566.rknn',
                  encoder_manifest='encoder-ZH_MIX_EN.fixed.onnx.inputs.json',
                  decoder_manifest='decoder-ZH_MIX_EN.fixed.onnx.inputs.json',
                  config_path='config.json',
                  mode='sim', target='rk3566')        # WSL 仿真
    audio = tts.tts(text=text, rng=np.random.RandomState(1234))  # 固定噪声便于复现
    sf.write("out_rknn.wav", audio, tts.sampling_rate)
    print("已生成 out_rknn.wav")

    # 可选: 调 Windows 播放 (同你原文件)
    p = '/mnt/c/Users/tyt/PyCharmMiscProject/com/robot/models/tts/out_rknn.wav'
    w = subprocess.check_output(['wslpath', '-w', p]).decode().strip()
    subprocess.run(['powershell.exe', '-c', f'(New-Object Media.SoundPlayer "{w}").PlaySync()'])