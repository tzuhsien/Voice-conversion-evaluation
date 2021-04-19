#!/usr/bin/env python
"""Preprocess script"""

import os
import json
import importlib
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import chain
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from librosa.util import find_files
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-w", "--n_workers", type=int, default=cpu_count())
    parser.add_argument("-a", "--audio_processor_path", type=str, required=True)
    return vars(parser.parse_args())


def load_process_save(audioprocessor, audio_path, save_dir):
    """Load an audio file, process, and save npz object."""
    wav, mel = audioprocessor.file2spectrogram(audio_path, return_wav=True)

    fd, temp_file_path = mkstemp(suffix=".npz", prefix="utterance-", dir=save_dir)
    np.savez_compressed(temp_file_path, wav=wav, mel=mel)
    os.close(fd)

    return {
        "feature_path": Path(temp_file_path).name,
        "audio_path": audio_path,
        "wav_len": len(wav),
        "mel_len": len(mel),
    }


def main(data_dirs, out_dir, n_workers, audio_processor_path):
    """Preprocess audio files into features for training."""

    audio_paths = chain.from_iterable([find_files(data_dir) for data_dir in data_dirs])

    audio_processor_path = Path(audio_processor_path) / "audioprocessor"
    audio_processor_path = str(audio_processor_path).replace("/", ".")
    audioprocessor = getattr(
        importlib.import_module(audio_processor_path), "AudioProcessor"
    )

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=n_workers)

    futures = []
    for audio_path in audio_paths:
        futures.append(
            executor.submit(load_process_save, audioprocessor, audio_path, save_dir)
        )

    infos = {
        "sample_rate": audioprocessor.sample_rate,
        "hop_len": audioprocessor.hop_len,
        "n_mels": audioprocessor.n_mels,
        "utterances": [future.result() for future in tqdm(futures, ncols=0)],
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
