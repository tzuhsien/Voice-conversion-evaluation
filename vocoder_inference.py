#!/usr/bin/env python
"""Preprocess script"""

import warnings
import random
import json
import importlib
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
from librosa.util import find_files
from tqdm import tqdm
import torch


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-a", "--audio_processor_path", type=str, required=True)
    parser.add_argument("-v", "--vocoder_path", type=str, required=True)
    parser.add_argument("-n", "--nums", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    return vars(parser.parse_args())


def batch_inference(out_mels, vocoder, batch_size):
    """Vocoder inferences in batch."""
    out_wavs = []
    for i in tqdm(range(0, len(out_mels), batch_size)):
        right = i + batch_size if len(out_mels) - i >= batch_size else len(out_mels)
        out_wavs.extend(vocoder.generate(out_mels[i:right]))
    return out_wavs


def main(data_dir, out_dir, audio_processor_path, vocoder_path, nums, batch_size):
    """Preprocess audio files into features for training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_paths = find_files(data_dir)
    audio_processor_path = Path(audio_processor_path) / "audioprocessor"
    audio_processor_path = str(audio_processor_path).replace("/", ".")
    audioprocessor = getattr(
        importlib.import_module(audio_processor_path), "AudioProcessor"
    )
    print(f"[INFO]: audioprocessor is loaded from {str(audio_processor_path)}.")

    vocoder = torch.jit.load(vocoder_path).to(device)
    print(f"[INFO]: Vocoder is loaded from {vocoder_path}.")
    assert vocoder.sample_rate == audioprocessor.sample_rate
    random.seed(531)
    try:
        audio_paths = random.sample(audio_paths, nums)
    except ValueError:
        pass

    mels = []
    for audio_path in tqdm(audio_paths):
        mel = audioprocessor.file2spectrogram(audio_path, return_wav=False)
        mel = torch.from_numpy(mel).to(device)
        mels.append(mel)
    print(f"[INFO]: {len(audio_paths)} audios is loaded.")

    with torch.no_grad():
        waveforms = batch_inference(mels, vocoder, batch_size=batch_size)

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, (audio_path, waveform) in enumerate(zip(audio_paths, waveforms)):
        output_path = save_dir / f"{i:04d}.wav"
        metadata.append({"original": audio_path, "converted": str(output_path)})
        waveform = waveform.detach().cpu().numpy()
        sf.write(output_path, waveform, vocoder.sample_rate)

    with open(save_dir / "metadata.json", "w") as file_pointer:
        json.dump(metadata, file_pointer, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
