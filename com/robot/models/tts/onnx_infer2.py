import os
import subprocess

import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import numpy as np
import onnxruntime as ort

from melo import utils
from melo.split_utils import split_sentence
import soundfile as sf
from melo.api import TTS



class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def sequence_mask(lengths, max_length=None):
    """
    torch: commons.sequence_mask
    lengths: [b] -> bool mask [b, max_length]
    """
    lengths = np.asarray(lengths)
    if max_length is None:
        max_length = int(lengths.max())
    x = np.arange(max_length)
    return (x[None, :] < lengths[:, None]).astype(np.float32)


class OnnxTTS:
    def __init__(self,
                 language,
                 encoder_path=None,
                 decoder_path=None,
                 config_path=None,
                 symbol_to_id=None,
                 device='cpu'):
        # ---------- 加载配置 (同 TTS.__init__) ----------
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        config = json.loads(data)

        hparams = HParams(**config)
        symbols = hparams.symbols

        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hparams
        self.symbol_to_id = symbol_to_id
        self.sampling_rate = hparams.data.sampling_rate

        self.encoder_sess = ort.InferenceSession(
            encoder_path, providers=['CPUExecutionProvider'])
        self.decoder_sess = ort.InferenceSession(
            decoder_path, providers=['CPUExecutionProvider'])

        # 读取固定的 x_length (导出时 x_length=256, 静态shape)
        self.x_length = 256

    def split_sentences_into_pieces(self, text, language):
        texts = split_sentence(text, language_str=language)
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
        return texts

    # =================================================================
    # 单句推理 — 与 export_onnx 内部计算流完全一致
    # =================================================================
    def prepare_input(self,
                      text,
                      language,
                      speaker_id):
        # ================= 1. 文本前端 (同 TTS.tts_to_file) =================
        if language in ['EN', 'ZH_MIX_EN']:
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, language, self.hps, 'cpu', self.symbol_to_id)
        # 转为 numpy
        x = phones.numpy().astype(np.int64).reshape(1, -1)
        x_lengths = np.array([phones.size(0)], dtype=np.int64)
        sid = np.array([speaker_id], dtype=np.int64).reshape(1)

        tone = tones.numpy().astype(np.int64).reshape(1, -1)  # [T]
        lang_ids = lang_ids.numpy().astype(np.int64).reshape(1, -1)  # [T]
        ja_bert = ja_bert.numpy().astype(np.float32)  # [768, T]
        if len(ja_bert.shape) == 2:
            ja_bert = ja_bert[np.newaxis, :, :]
        ja_bert = ja_bert.transpose(0, 2, 1) if ja_bert.shape[1] != 768 else ja_bert
        return x, x_lengths, sid, tone, lang_ids, ja_bert

    def audio_numpy_concat(self, segment_data_list, sr, speed=1.0):
        audio_segment_res = []
        for segment_data in segment_data_list:
            audio_segment_res += segment_data.reshape(-1).tolist()
            audio_segment_res += [0] * int((sr * 0.05) / speed)
        audio_segment_res = np.array(audio_segment_res).astype(np.float32)
        return audio_segment_res

    # def pad_2d(self, arr, target_len):
    #     pad_len = target_len - arr.shape[1]
    #     if pad_len > 0:
    #         padding = np.zeros((arr.shape[0], pad_len), dtype=arr.dtype)
    #         arr = np.concatenate([arr, padding], axis=1)
    #     return arr
    #
    # def pad_3d(self, arr, target_len):
    #     pad_len = target_len - arr.shape[2]
    #     if pad_len > 0:
    #         padding = np.zeros((arr.shape[0], arr.shape[1], pad_len), dtype=arr.dtype)
    #         arr = np.concatenate([arr, padding], axis=2)
    #     return arr

    def pad_2d(self, arr, target_len):
        """用最后一个有效元素复制填充，避免内部 reflect padding 被 0 污染"""
        pad_len = target_len - arr.shape[1]
        if pad_len > 0:
            # arr shape: [1, T] -> 取最后一列并复制 pad_len 次
            padding = np.repeat(arr[:, -1:], pad_len, axis=1)
            arr = np.concatenate([arr, padding], axis=1)
        return arr

    def pad_3d(self, arr, target_len):
        """用最后一个有效元素复制填充"""
        pad_len = target_len - arr.shape[2]
        if pad_len > 0:
            # arr shape: [1, C, T] -> 取最后一帧并复制 pad_len 次
            padding = np.repeat(arr[:, :, -1:], pad_len, axis=2)
            arr = np.concatenate([arr, padding], axis=2)
        return arr

    def tts(self, text, speaker_id=1, language='ZH_MIX_EN', sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0):
        texts = self.split_sentences_into_pieces(text, language)
        audio_list = []
        x_max_len = 256
        for t in texts:
            x, x_lengths, sid, tone, lang_ids, ja_bert = self.prepare_input(t, language, speaker_id)
            ja_bert = self.pad_3d(ja_bert, x_max_len)
            lang_ids = self.pad_2d(lang_ids, x_max_len)
            tone = self.pad_2d(tone, x_max_len)
            x = self.pad_2d(x, x_max_len)

            encoder_inputs = {
                'x': x,
                'x_lengths': np.array([x_lengths[0]]),
                'sid': sid,
                'tone': tone,
                'lang_ids': lang_ids,
                'ja_bert': ja_bert,
                'noise_scale_w': np.array([noise_scale_w], dtype=np.float32),
                'sdp_ratio': np.array([sdp_ratio], dtype=np.float32),
            }
            encoder_outputs = self.encoder_sess.run(
                None,
                encoder_inputs
            )
            logw, x_mask, g, m_p, logs_p = encoder_outputs

            # ================= 5. Decoder ONNX =================
            length_scale = 1.0 / speed
            w = np.exp(logw) * x_mask * length_scale
            w_ceil = np.ceil(w)
            y_length_max = 2 * x_max_len
            y_mask = np.zeros((1, 1, y_length_max), dtype=np.float32)
            y_mask[:, :, :] = 1.0
            attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
            attn = self.generate_path(w_ceil, attn_mask).squeeze(1)
            #  对应: self.model.forward_decoder(attn, y_mask, g, m_p, logs_p, noise_scale)
            decoder_inputs = {
                'attn': attn.astype(np.float32),  # (1, 512, 256) float32
                'y_mask': y_mask.astype(np.float32),  # (1, 1, 512)   float32
                'g': g,  # (1, 512, 1)   float32
                'm_p': m_p,  # (1, 192, 256) float32
                'logs_p': logs_p,  # (1, 192, 256) float32
                'noise_scale': np.array([noise_scale], dtype=np.float32)  # (1,)          float32
            }
            decoder_outputs = self.decoder_sess.run(None, decoder_inputs)

            # ================= 6. 截取有效音频 =================
            #  有效帧数 = sum(w_ceil) (仅统计有效部分)
            #  简化: 用全部帧长(静态导出固定为512帧), 也可以按实际duration截取
            audio = decoder_outputs[0][0, 0, :]

            # # 计算实际需要的帧数用于截取
            total_frames = int(np.sum(w_ceil * x_mask))  # 有效帧数
            if 0 < total_frames < audio.shape[0]:
                # 估算每帧采样点数
                samples_per_frame = audio.shape[0] // y_mask.shape[2]
                audio = audio[:total_frames * samples_per_frame]
            audio_list.append(audio)
        return self.audio_numpy_concat(audio_list, sr=self.sampling_rate, speed=speed)

    def generate_path(self, duration, mask):
        """
        torch: commons.generate_path
        duration: [b, 1, t_x]      例: (1, 1, 256)
        mask:     [b, 1, t_y, t_x] 例: (1, 1, 512, 256)
        return:   [b, 1, t_y, t_x] 例: (1, 1, 512, 256)
        """
        b, _, t_y, t_x = mask.shape

        # torch: cum_duration = torch.cumsum(duration, -1)        # [b, 1, t_x]
        cum_duration = np.cumsum(duration, axis=-1)  # [b, 1, t_x]

        # torch: cum_duration_flat = cum_duration.view(b * t_x)   # [b*t_x]
        cum_duration_flat = cum_duration.reshape(b * t_x)  # [b*t_x]

        path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)

        # torch: path = path.view(b, t_x, t_y)                    # [b, t_x, t_y]
        path = path.reshape(b, t_x, t_y)  # [b, t_x, t_y]

        # torch: path = path - F.pad(path, [[0,0],[1,0],[0,0]])[:, :-1]
        #   等价于在 t_x 维度前向差分
        padded = np.pad(path, ((0, 0), (1, 0), (0, 0)), mode='constant')
        path = path - padded[:, :-1, :]  # [b, t_x, t_y]

        # torch: path = path.unsqueeze(1).transpose(2,3) * mask
        #   path.unsqueeze(1)          -> [b, 1, t_x, t_y]
        #   .transpose(2,3)            -> [b, 1, t_y, t_x]
        path = path[:, None].transpose(0, 1, 3, 2) * mask  # [b, 1, t_y, t_x]

        return path



# Speed is adjustable
if __name__ == '__main__':
    # import nltk
    # nltk.download('averaged_perceptron_tagger_eng')
    # 加载
    text = "我最近在学习machine learning。前三个隔热"
    speed = 1.0
    device = 'cpu'  # or cuda:0
    config_path = 'config.json'


    tts = OnnxTTS(language='ZH', encoder_path='encoder-ZH_MIX_EN.onnx', decoder_path='decoder-ZH_MIX_EN.onnx',
                  config_path = config_path, device='cpu')

    # 合成
    audio_data = tts.tts(text=text)
    sf.write("out0830.wav", audio_data,tts.hps.data.sampling_rate)

    # # # 1. 生成 WAV 文件到 WSL 可访问的 Windows 路径（例如 /mnt/c/...）
    output_path_wsl = '/mnt/c/Users/tyt/PyCharmMiscProject/com/robot/models/tts/out0830.wav'
    #
    # # 2. 将 WSL 路径转换为 Windows 路径
    windows_path = subprocess.check_output(['wslpath', '-w', output_path_wsl]).decode().strip()
    # # 例如得到 C:\Users\tyt\...\zh.wav
    #
    # # 3. 使用 PowerShell 播放（阻塞直到播放完成）
    subprocess.run(['powershell.exe', '-c', f'(New-Object Media.SoundPlayer "{windows_path}").PlaySync()'])

