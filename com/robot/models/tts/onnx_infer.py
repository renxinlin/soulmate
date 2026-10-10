# import os
# import re
# import json
# import torch
# import librosa
# import soundfile
# import torchaudio
# import numpy as np
# import torch.nn as nn
# from tqdm import tqdm
# import onnxruntime as ort
#
# from api import TTS
# from melo.text import language_id_map
# from melo import utils
# from melo import commons
# from melo.split_utils import split_sentence
# from melo.mel_processing import spectrogram_torch, spectrogram_torch_conv
# from melo.download_utils import load_or_download_config, load_or_download_model
#
#
# def _sequence_mask(length, max_length=None):
#     """numpy 实现的 commons.sequence_mask"""
#     if max_length is None:
#         max_length = length.max()
#     x = np.arange(max_length, dtype=length.dtype)
#     return x[np.newaxis, :] < length[:, np.newaxis]
#
#
# def _generate_path(duration, mask):
#     """numpy 实现的 commons.generate_path"""
#     b, _, t_y, t_x = mask.shape
#     cum_duration = np.cumsum(duration, axis=-1)
#     cum_duration_flat = cum_duration.reshape(b * t_x)
#     path = _sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
#     path = path.reshape(b, t_x, t_y)
#     path_padded = np.pad(path, ((0, 0), (1, 0), (0, 0)), mode='constant')
#     path = path - path_padded[:, :-1, :]
#     path = np.expand_dims(path, 1).transpose(0, 1, 3, 2) * mask
#     return path
#
#
# class TTSOnnx(nn.Module):
#     def __init__(self,
#                  language,
#                  encoder_path,
#                  decoder_path,
#                  device='cpu',
#                  use_hf=True,
#                  hps=None):
#         super().__init__()
#         self.hps = hps
#         self.symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}
#         self.device = device
#
#         language = language.split('_')[0]
#         self.language = 'ZH_MIX_EN' if language == 'ZH' else language
#
#         # 加载 ONNX 模型
#         self.encoder_session = ort.InferenceSession(encoder_path)
#         self.decoder_session = ort.InferenceSession(decoder_path)
#
#         self.encoder_input_names = [inp.name for inp in self.encoder_session.get_inputs()]
#         self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
#
#     @staticmethod
#     def audio_numpy_concat(segment_data_list, sr, speed=1.):
#         audio_segments = []
#         for segment_data in segment_data_list:
#             audio_segments += segment_data.reshape(-1).tolist()
#             audio_segments += [0] * int((sr * 0.05) / speed)
#         audio_segments = np.array(audio_segments).astype(np.float32)
#         return audio_segments
#
#     @staticmethod
#     def split_sentences_into_pieces(text, language, quiet=False):
#         texts = split_sentence(text, language_str=language)
#         if not quiet:
#             print(" > Text split to sentences.")
#             print('\n'.join(texts))
#             print(" > ===========================")
#         return texts
#
#     def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
#         language = self.language
#         texts = self.split_sentences_into_pieces(text, language, quiet)
#         audio_list = []
#         if pbar:
#             tx = pbar(texts)
#         else:
#             if position:
#                 tx = tqdm(texts, position=position)
#             elif quiet:
#                 tx = texts
#             else:
#                 tx = tqdm(texts)
#         for t in tx:
#             if language in ['EN', 'ZH_MIX_EN']:
#                 t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
#             device = self.device
#
#             # 文本特征提取，与原模型完全一致
#             bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
#
#             # ---------- Encoder 输入 ----------
#             x = phones.cpu().numpy().astype(np.int64)
#             x_lengths = np.array([phones.size(0)], dtype=np.int64)
#             tones_np = tones.cpu().numpy().astype(np.int64)
#             lang_ids_np = lang_ids.cpu().numpy().astype(np.int64)
#             ja_bert_np = ja_bert.cpu().numpy().astype(np.float32)
#
#             if x.ndim == 1:
#                 x = np.expand_dims(x, 0)
#             if tones_np.ndim == 1:
#                 tones_np = np.expand_dims(tones_np, 0)
#             if lang_ids_np.ndim == 1:
#                 lang_ids_np = np.expand_dims(lang_ids_np, 0)
#             if ja_bert_np.ndim == 2:
#                 ja_bert_np = np.expand_dims(ja_bert_np, 0)
#
#             sid = np.array([speaker_id], dtype=np.int64)
#             noise_scale_w_np = np.array([noise_scale_w], dtype=np.float32)
#             sdp_ratio_np = np.array([sdp_ratio], dtype=np.float32)
#             length_scale = 1. / speed
#             noise_scale_np = np.array([noise_scale], dtype=np.float32)
#
#             # ---------- 运行 Encoder ----------
#             encoder_inputs = {
#                 'x': x,
#                 'x_lengths': x_lengths,
#                 'sid': sid,
#                 'tone': tones_np,
#                 'lang_ids': lang_ids_np,
#                 'ja_bert': ja_bert_np,
#                 'noise_scale_w': noise_scale_w_np,
#                 'sdp_ratio': sdp_ratio_np,
#             }
#             encoder_inputs = {k: v for k, v in encoder_inputs.items() if k in self.encoder_input_names}
#             logw, x_mask, g, m_p, logs_p = self.encoder_session.run(None, encoder_inputs)
#
#             # ---------- 手动计算 duration / attention ----------
#             w = np.exp(logw) * x_mask * length_scale
#             w_ceil = np.ceil(w)
#             y_lengths = w_ceil.sum(axis=(1, 2)).astype(np.int64)
#             y_lengths = np.maximum(y_lengths, 1)
#
#             y_mask = _sequence_mask(y_lengths, y_lengths.max()).astype(x_mask.dtype)
#             y_mask = np.expand_dims(y_mask, 1)
#
#             attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
#             attn = _generate_path(w_ceil, attn_mask).squeeze(1)
#
#             # ---------- 运行 Decoder ----------
#             decoder_inputs = {
#                 'attn': attn,
#                 'y_mask': y_mask,
#                 'g': g,
#                 'm_p': m_p,
#                 'logs_p': logs_p,
#                 'noise_scale': noise_scale_np,
#             }
#             decoder_inputs = {k: v for k, v in decoder_inputs.items() if k in self.decoder_input_names}
#             y = self.decoder_session.run(None, decoder_inputs)[0]
#
#             audio = y[0, 0].astype(np.float32)
#             audio_list.append(audio)
#
#         torch.cuda.empty_cache()
#         audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
#
#         if output_path is None:
#             return audio
#         else:
#             if format:
#                 soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
#             else:
#                 soundfile.write(output_path, audio, self.hps.data.sampling_rate)
#
# if __name__ == '__main__':
#     model_pth = TTS(language='ZH', device='cpu')
#
#     model = TTSOnnx(
#         language="ZH",
#         encoder_path="encoder-ZH_MIX_EN.onnx",
#         decoder_path="decoder-ZH_MIX_EN.onnx",
#         hps=model_pth.hps
#     )
#
#     audio = model.tts_to_file(
#         text="你好，这是 ONNX 推理测试。",
#         speaker_id=0,
#         output_path="output.wav",
#         sdp_ratio=0.2,
#         noise_scale=0.6,
#         noise_scale_w=0.8,
#         speed=1.0
#     )