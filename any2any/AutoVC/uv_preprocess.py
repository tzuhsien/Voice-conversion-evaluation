#!/usr/bin/env python
"""Preprocess script"""

import os
import json
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import chain
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from librosa.util import find_files
from tqdm import tqdm

from modules.audioprocessor import AudioProcessor


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-w", "--n_workers", type=int, default=cpu_count())
    return vars(parser.parse_args())


def load_process_save(audio_path, save_dir):
    """Load an audio file, process, and save npz object."""
    wav, mel = AudioProcessor.file2spectrogram(audio_path, return_wav=True)

    fd, temp_file_path = mkstemp(suffix=".npz", prefix="utterance-", dir=save_dir)
    np.savez_compressed(temp_file_path, wav=wav, mel=mel)
    os.close(fd)

    return {
        "feature_path": Path(temp_file_path).name,
        "audio_path": audio_path,
        "wav_len": len(wav),
        "mel_len": len(mel),
    }


def main(data_dirs, out_dir, n_workers):
    """Preprocess audio files into features for training."""

    audio_paths = chain.from_iterable([find_files(data_dir) for data_dir in data_dirs])

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=n_workers)

    futures = []
    for audio_path in audio_paths:
        futures.append(executor.submit(load_process_save, audio_path, save_dir))

    infos = {
        "sample_rate": AudioProcessor.sample_rate,
        "hop_len": AudioProcessor.hop_len,
        "n_mels": 80,
        "utterances": [future.result() for future in tqdm(futures, ncols=0)],
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
