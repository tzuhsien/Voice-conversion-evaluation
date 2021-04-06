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

from preprocess.tacotron.hyperparams import Hyperparams as hp
from preprocess.tacotron.utils import load_wav, wav2melspectrogram


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-w", "--n_workers", type=int, default=cpu_count())
    return vars(parser.parse_args())


def load_process_save(audio_path, save_dir):
    """Load an audio file, process, and save npz object."""
    wav = load_wav(audio_path)
    mel = wav2melspectrogram(wav)

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
        "sample_rate": hp.sr,
        "hop_len": hp.hop_length,
        "n_mels": hp.n_mels,
        "utterances": [future.result() for future in tqdm(futures, ncols=0)],
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
