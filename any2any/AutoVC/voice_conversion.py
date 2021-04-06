#!/usr/bin/env python3
import argparse
import warnings
from math import ceil
from collections import OrderedDict

import numpy as np
import torch
import soundfile as sf

from modules.model_bl import D_VECTOR
from modules.model_vc import Generator
from modules.audioprocessor import AudioProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


def file2mel(fpath):
    return AudioProcessor.file2spectrogram(fpath)


def encode_uttr(encoder, mel):
    melsp = torch.from_numpy(mel[np.newaxis, :, :]).to(device)
    emb = encoder(melsp)
    return emb


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad


def conversion(generator, src_emb, tar_emb, uttr):
    x_org, len_pad = pad_seq(uttr)
    uttr = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    src_emb = src_emb.unsqueeze(0)
    tar_emb = tar_emb.unsqueeze(0)
    _, x_identic_psnt, _ = generator(uttr, src_emb, tar_emb)
    if len_pad == 0:
        result = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        result = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    return result


def main(source, targets, output, encoder_path, generator_path, vocoder_path):
    # file to mel-spectrogram
    src_mel = file2mel(source)
    tar_mels = [file2mel(target) for target in targets]

    # speaker encoding
    encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    ckpt = torch.load(encoder_path, map_location=device)
    state_dict = OrderedDict()
    for key, val in ckpt["model_b"].items():
        state_dict[key[7:]] = val
    encoder.load_state_dict(state_dict)
    src_emb = encode_uttr(encoder, src_mel).squeeze()
    tar_embs = [encode_uttr(encoder, tar_mel).squeeze() for tar_mel in tar_mels]
    tar_emb = torch.stack(tar_embs).mean(dim=0)
    # meta: [[emb_1, [mel_1a, mel_1b, ...]], [emb_2, [mel_2a, mel_2b, ...]], ...]

    # voice conversion
    generator = Generator(32, 256, 512, 32).eval().to(device)
    ckpt = torch.load(generator_path, map_location=device)
    generator.load_state_dict(ckpt["model"])
    result_mel = conversion(generator, src_emb, tar_emb, src_mel)

    # convert mel-spectrogram into waveform
    vocoder = torch.jit.load(vocoder_path).to(device)
    mel_tensor = torch.from_numpy(result_mel).unsqueeze(0).to(device)
    wav_tensor = vocoder.generate(mel_tensor)
    wav_array = wav_tensor.squeeze().cpu().numpy()
    sf.write(output, wav_array.astype(np.float32), vocoder.sample_rate)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("targets", nargs="+", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("-e", "--encoder_path", type=str, default="models/encoder.ckpt")
    parser.add_argument(
        "-g", "--generator_path", type=str, default="models/autovc.ckpt"
    )
    parser.add_argument(
        "-v", "--vocoder_path", type=str, default="models/vocoder-ckpt-90000.pt"
    )

    args = parser.parse_args()
    with torch.no_grad():
        main(**vars(args))
