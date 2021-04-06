#!/usr/bin/env python3
import os
import argparse
import warnings
from math import ceil
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm
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


def sort_func(element):
    return element[0].shape[0]


def batch_inference(out_mels, vocoder, batch_size):
    out_wavs = []
    for i in tqdm(range(0, len(out_mels), batch_size)):
        right = i + batch_size if len(out_mels) - i >= batch_size else len(out_mels)
        out_wavs.extend(vocoder.generate(out_mels[i:right]))
    return out_wavs


def main(
    pairs_path,
    source_dir,
    target_dir,
    output_dir,
    encoder_path,
    generator_path,
    vocoder_path,
    use_vocoder,
):
    # file to mel-spectrogram
    pairs = torch.load(pairs_path)

    # speaker encoding
    encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    ckpt = torch.load(encoder_path, map_location=device)
    state_dict = OrderedDict()
    for key, val in ckpt["model_b"].items():
        state_dict[key[7:]] = val
    encoder.load_state_dict(state_dict)

    # generator
    generator = Generator(32, 256, 512, 32).eval().to(device)
    ckpt = torch.load(generator_path, map_location=device)
    generator.load_state_dict(ckpt["model"])

    # vocoder
    vocoder = torch.jit.load(vocoder_path).to(device)

    mel_tensors = []
    if not os.path.exists(
        os.path.join(output_dir, "out_mels.tar")
    ) or not os.path.exists(os.path.join(output_dir, "pairs.tar")):
        for pair in tqdm(pairs):
            try:
                src_mel = file2mel(os.path.join(source_dir, pair["src_utt"]))
            except FileNotFoundError:
                src_mel = file2mel(os.path.join(source_dir, "wav", pair["src_utt"]))

            tar_mels = [
                file2mel(os.path.join(target_dir, target)) for target in pair["tgt_utt"]
            ]

            src_emb = encode_uttr(encoder, src_mel).squeeze()
            tar_embs = [encode_uttr(encoder, tar_mel).squeeze() for tar_mel in tar_mels]
            tar_emb = torch.stack(tar_embs).mean(dim=0)
            # meta: [[emb_1, [mel_1a, mel_1b, ...]], [emb_2, [mel_2a, mel_2b, ...]], ...]

            # voice conversion
            result_mel = conversion(generator, src_emb, tar_emb, src_mel)

            # convert mel-spectrogram into waveform
            mel_tensor = torch.from_numpy(result_mel).to(device)
            mel_tensors.append(mel_tensor)
        zipped = zip(mel_tensors, pairs)
        zip_sorted = sorted(zipped, key=sort_func, reverse=True)
        tuples = zip(*zip_sorted)
        mel_tensors, pairs = [list(i) for i in tuples]
        torch.save(mel_tensors, os.path.join(output_dir, "out_mels.tar"))
        torch.save(pairs, os.path.join(output_dir, "pairs.tar"))
    else:
        mel_tensors = torch.load(os.path.join(output_dir, "out_mels.tar"))
        pairs = torch.load(os.path.join(output_dir, "pairs.tar"))

    if use_vocoder == False:
        return
    with torch.no_grad():
        wav_tensors = batch_inference(mel_tensors, vocoder, batch_size=20)

    for i, pair in tqdm(enumerate(pairs)):
        wav_array = wav_tensors[i].cpu().numpy()
        prefix = Path(pair["src_utt"]).name.replace("/wav/", "_").replace(".wav", "")
        postfix = (
            Path(pair["tgt_utt"][0]).name.replace("/wav/", "_").replace(".wav", "")
        )
        sf.write(
            os.path.join(output_dir, f"{prefix}_to_{postfix}.wav"),
            wav_array.astype(np.float32),
            vocoder.sample_rate,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_path", type=str)
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-e", "--encoder_path", type=str, default="models/encoder.ckpt")
    parser.add_argument(
        "-g", "--generator_path", type=str, default="models/autovc.ckpt"
    )
    parser.add_argument(
        "-v",
        "--vocoder_path",
        type=str,
        default="./models/parallel-vocoder-ckpt-150000.pt",
    )
    parser.add_argument("--use_vocoder", type=bool, default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        main(**vars(args))
