import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import re
import numpy as np
import soundfile as sf
from melo import utils
from melo.split_utils import split_sentence


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
    """与 torch 版本相同的 sequence_mask 实现"""
    lengths = np.asarray(lengths)
    if max_length is None:
        max_length = int(lengths.max())
    x = np.arange(max_length)
    return (x[None, :] < lengths[:, None]).astype(np.float32)


class RknnTTS:
    def __init__(self,
                 language,
                 encoder_path='encoderk3566.rknn',
                 decoder_path='decoderk3566.rknn',
                 config_path='config.json',
                 device='simulator'):   # 'simulator' 或 'rk3566'
        # ---------- 加载配置 ----------
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        hparams = HParams(**config)
        self.hps = hparams
        self.symbol_to_id = {s: i for i, s in enumerate(hparams.symbols)}
        self.sampling_rate = hparams.data.sampling_rate
        self.x_length = 256   # 固定长度，导出 RKNN 时确定

        # ---------- 初始化 RKNN 模型 ----------
        from rknn.api import RKNN

        # 创建 encoder RKNN 对象
        self.encoder_rknn = RKNN(verbose=True)
        print(f'--> Loading encoder RKNN model: {encoder_path}')
        ret = self.encoder_rknn.load_rknn(encoder_path)
        self.encoder_rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=None)

        if ret != 0:
            raise RuntimeError(f'Load encoder RKNN failed! ret = {ret}')
        print(f'--> Init encoder runtime on {device}')
        ret = self.encoder_rknn.init_runtime(target=device)
        if ret != 0:
            raise RuntimeError(f'Init encoder runtime failed! ret = {ret}')

        # 创建 decoder RKNN 对象
        self.decoder_rknn = RKNN(verbose=True)
        print(f'--> Loading decoder RKNN model: {decoder_path}')
        ret = self.decoder_rknn.load_rknn(decoder_path)
        self.decoder_rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=None)
        if ret != 0:
            raise RuntimeError(f'Load decoder RKNN failed! ret = {ret}')
        print(f'--> Init decoder runtime on {device}')
        ret = self.decoder_rknn.init_runtime(target=device)
        if ret != 0:
            raise RuntimeError(f'Init decoder runtime failed! ret = {ret}')

    def split_sentences_into_pieces(self, text, language):
        texts = split_sentence(text, language_str=language)
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
        return texts

    def prepare_input(self, text, language, speaker_id):
        """文本预处理，返回 int32 类型的输入"""
        if language in ['EN', 'ZH_MIX_EN']:
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, language, self.hps, 'cpu', self.symbol_to_id)

        # RKNN 不支持 int64，统一使用 int32
        int_dtype = np.int32
        x = phones.numpy().astype(int_dtype).reshape(1, -1)
        x_lengths = np.array([phones.size(0)], dtype=int_dtype)
        sid = np.array([speaker_id], dtype=int_dtype).reshape(1)

        tone = tones.numpy().astype(int_dtype).reshape(1, -1)
        lang_ids = lang_ids.numpy().astype(int_dtype).reshape(1, -1)
        ja_bert = ja_bert.numpy().astype(np.float32)
        if len(ja_bert.shape) == 2:
            ja_bert = ja_bert[np.newaxis, :, :]
        ja_bert = ja_bert.transpose(0, 2, 1) if ja_bert.shape[1] != 768 else ja_bert
        return x, x_lengths, sid, tone, lang_ids, ja_bert

    def audio_numpy_concat(self, segment_data_list, sr, speed=1.0):
        audio_segment_res = []
        for segment_data in segment_data_list:
            audio_segment_res += segment_data.reshape(-1).tolist()
            audio_segment_res += [0] * int((sr * 0.05) / speed)
        return np.array(audio_segment_res).astype(np.float32)

    def pad_2d(self, arr, target_len):
        """用最后一个有效元素复制填充，避免 reflect padding 被 0 污染"""
        pad_len = target_len - arr.shape[1]
        if pad_len > 0:
            padding = np.repeat(arr[:, -1:], pad_len, axis=1)
            arr = np.concatenate([arr, padding], axis=1)
        return arr

    def pad_3d(self, arr, target_len):
        """用最后一个有效元素复制填充"""
        pad_len = target_len - arr.shape[2]
        if pad_len > 0:
            padding = np.repeat(arr[:, :, -1:], pad_len, axis=2)
            arr = np.concatenate([arr, padding], axis=2)
        return arr

    def tts(self, text, speaker_id=1, language='ZH_MIX_EN',
            sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0):
        texts = self.split_sentences_into_pieces(text, language)
        audio_list = []
        x_max_len = self.x_length

        for t in texts:
            x, x_lengths, sid, tone, lang_ids, ja_bert = self.prepare_input(t, language, speaker_id)

            # 填充到固定长度
            ja_bert = self.pad_3d(ja_bert, x_max_len)
            lang_ids = self.pad_2d(lang_ids, x_max_len)
            tone = self.pad_2d(tone, x_max_len)
            x = self.pad_2d(x, x_max_len)

            # 确保所有输入数组连续（RKNN 要求）
            x = np.ascontiguousarray(x)
            x_lengths_arr = np.ascontiguousarray(np.array([x_lengths[0]], dtype=np.int32))
            sid = np.ascontiguousarray(sid)
            tone = np.ascontiguousarray(tone)
            lang_ids = np.ascontiguousarray(lang_ids)
            ja_bert = np.ascontiguousarray(ja_bert)
            noise_scale_w_arr = np.ascontiguousarray(np.array([noise_scale_w], dtype=np.float32))
            sdp_ratio_arr = np.ascontiguousarray(np.array([sdp_ratio], dtype=np.float32))

            # ---------- Encoder 推理 ----------
            encoder_inputs = [x, x_lengths_arr, sid, tone, lang_ids, ja_bert,
                              noise_scale_w_arr, sdp_ratio_arr]
            encoder_outputs = self.encoder_rknn.inference(inputs=encoder_inputs)
            logw, x_mask, g, m_p, logs_p = encoder_outputs

            # ---------- 计算 duration / attention ----------
            length_scale = 1.0 / speed
            w = np.exp(logw) * x_mask * length_scale
            w_ceil = np.ceil(w)
            y_length_max = 2 * x_max_len
            y_mask = np.zeros((1, 1, y_length_max), dtype=np.float32)
            y_mask[:, :, :] = 1.0
            attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
            attn = self.generate_path(w_ceil, attn_mask).squeeze(1)

            # ---------- Decoder 推理 ----------
            decoder_inputs = [
                np.ascontiguousarray(attn.astype(np.float32)),
                np.ascontiguousarray(y_mask.astype(np.float32)),
                np.ascontiguousarray(g),
                np.ascontiguousarray(m_p),
                np.ascontiguousarray(logs_p),
                np.ascontiguousarray(np.array([noise_scale], dtype=np.float32))
            ]
            decoder_outputs = self.decoder_rknn.inference(inputs=decoder_inputs)
            audio = decoder_outputs[0][0, 0, :]

            # 截取有效音频（与原始 ONNX 逻辑相同）
            total_frames = int(np.sum(w_ceil * x_mask))
            if 0 < total_frames < audio.shape[0]:
                samples_per_frame = audio.shape[0] // y_mask.shape[2]
                audio = audio[:total_frames * samples_per_frame]
            audio_list.append(audio)

        return self.audio_numpy_concat(audio_list, sr=self.sampling_rate, speed=speed)

    def generate_path(self, duration, mask):
        """生成 attention path（与原始 torch 实现一致）"""
        b, _, t_y, t_x = mask.shape
        cum_duration = np.cumsum(duration, axis=-1)
        cum_duration_flat = cum_duration.reshape(b * t_x)
        path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
        path = path.reshape(b, t_x, t_y)
        padded = np.pad(path, ((0, 0), (1, 0), (0, 0)), mode='constant')
        path = path - padded[:, :-1, :]
        path = path[:, None].transpose(0, 1, 3, 2) * mask
        return path

    def release(self):
        """释放 RKNN 资源"""
        self.encoder_rknn.release()
        self.decoder_rknn.release()


if __name__ == '__main__':
    text = "我最近在学习machine learning。前三个隔热"
    speed = 1.0
    config_path = 'config.json'

    # 选择模式：'simulator' 用于 PC 仿真，'rk3566' 用于真实硬件
    device = 'simulator'   # 也可以改为 'rk3566'

    tts = RknnTTS(language='ZH',
                  encoder_path='encoderk3566.rknn',
                  decoder_path='decoderk3566.rknn',
                  config_path=config_path,
                  device=device)

    audio_data = tts.tts(text=text)
    sf.write(f"out_{device}.wav", audio_data, tts.hps.data.sampling_rate)

    tts.release()