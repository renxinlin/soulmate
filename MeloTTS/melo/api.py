import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import onnx, onnxsim

from .text import language_id_map
from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

class TTS(nn.Module):
    def __init__(self,
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path =
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device

        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            print("---------------------")
            print(t)

            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
        return audio

    def export_onnx(self, speaker_id, x_length=256, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0,):
        language = self.language
        lang_id = language_id_map[language]

        # Export the model
        with torch.no_grad():
            x_lengths = torch.LongTensor([x_length])
            bert = torch.zeros(1, 1024, x_length)
            ja_bert = torch.zeros(1, 768, x_length)
            x = torch.zeros(x_length, dtype=torch.int64).unsqueeze(0)
            tone = torch.randint(1, 5, size=(x_length,), dtype=torch.int64).unsqueeze(0)
            lang_ids = torch.zeros_like(x)
            lang_ids[:, 1::2] = lang_id
            noise_scale = torch.FloatTensor([noise_scale])
            noise_scale_w = torch.FloatTensor([noise_scale_w])
            length_scale = torch.FloatTensor([1./speed])
            sdp_ratio = torch.FloatTensor([sdp_ratio])
            sid = torch.LongTensor([speaker_id])

            logw, x_mask, g, m_p, logs_p = self.model.forward_encoder(
                    x,
                    x_lengths,
                    sid,
                    tone,
                    lang_ids,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale_w=noise_scale_w,
                )
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.FloatTensor([2 * x_length])
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                x_mask.dtype
            )
            # x_mask (1, 1, 256)         y_mask=  (1 1 512) attn_mask = （1,1,512,256）
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask).squeeze(1)

            # Export the encoder model
            inputs = (
                x, x_lengths, sid, tone, lang_ids, ja_bert, noise_scale_w, sdp_ratio
            )
            input_names = ['x', 'x_lengths', 'sid', 'tone', 'lang_ids', 'ja_bert', 'noise_scale_w', 'sdp_ratio']
            encoder_name = f"encoder-{language}.onnx"
            self.model.forward = self.model.forward_encoder
            torch.onnx.export(self.model,
                            inputs,
                            encoder_name,
                            opset_version=16,
                            input_names = input_names,
                            output_names = ["logw", "x_mask", "g", "m_p", "logs_p"],
                            )
            # sim_model,_ = onnxsim.simplify(encoder_name)
            # onnx.save(sim_model, encoder_name)
            print(f"Export encoder to {encoder_name}")

            # export decoder
            inputs = (
                attn, y_mask, g, m_p, logs_p, noise_scale
            )
            decoder_name = f"decoder-{language}.onnx"
            self.model.forward = self.model.forward_decoder
            torch.onnx.export(
                                self.model,
                                inputs,
                                decoder_name,
                                export_params=True,
                                opset_version=16,
                                # do_constant_folding=True,
                                input_names = ["attn", "y_mask", "g", "m_p", "logs_p", "noise_scale"],
                                output_names = ['y']
                            )
            # sim_model,_ = onnxsim.simplify(decoder_name)
            # onnx.save(sim_model, decoder_name)
            print(f"Export decoder to {decoder_name}")