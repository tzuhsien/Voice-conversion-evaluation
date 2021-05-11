#!/usr/bin/env python
"""Preprocess script"""

import warnings
import json
from importlib import import_module
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
from tqdm import tqdm
import torch


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-a", "--audio_processor_path", type=str, required=True)
    parser.add_argument("-v", "--vocoder_path", type=str, required=True)
    parser.add_argument("-n", "--nums", type=int, default=2000)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument("-c", "--corpus_name", type=str)
    parser.add_argument("-p", "--parser_dir", type=str)
    return vars(parser.parse_args())


def batch_inference(out_mels, vocoder, batch_size):
    """Vocoder inferences in batch."""
    out_wavs = []
    for i in tqdm(range(0, len(out_mels), batch_size)):
        right = i + batch_size if len(out_mels) - i >= batch_size else len(out_mels)
        out_wavs.extend(vocoder.generate(out_mels[i:right]))
    # for i in tqdm(range(len(out_mels))):
    #     out_wavs.append(vocoder.generate(out_mels[i].unsqueeze(0)))
    return out_wavs


def main(
    data_dir,
    out_dir,
    audio_processor_path,
    vocoder_path,
    nums,
    batch_size,
    corpus_name,
    parser_dir,
):
    """Preprocess audio files into features for training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_processor_path = Path(audio_processor_path) / "audioprocessor"
    audio_processor_path = str(audio_processor_path).replace("/", ".")
    audioprocessor = getattr(import_module(audio_processor_path), "AudioProcessor")
    print(f"[INFO]: audioprocessor is loaded from {str(audio_processor_path)}.")

    vocoder = torch.jit.load(vocoder_path).to(device)
    print(f"[INFO]: Vocoder is loaded from {vocoder_path}.")
    assert vocoder.sample_rate == audioprocessor.sample_rate

    parser_path = str(Path(parser_dir) / f"{corpus_name}_parser").replace("/", ".")
    Parser = getattr(import_module(parser_path), "Parser")
    parser = Parser(data_dir)
    seed = 5
    parser.set_random_seed(seed)
    metadata = {
        "model": vocoder_path,
        "audio_processor": audio_processor_path,
        "source_corpus": None,
        "target_corpus": corpus_name,
        "sample_number": nums,
        "target_number": 1,
        "source_random_seed": None,
        "target_random_seed": parser.seed,
        "pairs": [],
    }

    mels = []
    for _ in tqdm(range(nums)):
        audio_path, speaker_id, content, second = parser.sample_source()
        metadata["pairs"].append(
            {
                "source_speaker": None,
                "target_speaker": speaker_id,
                "src_utt": None,
                "tgt_utts": [audio_path],
                "content": content,
                "src_second": second
            }
        )
        audio_path = Path(data_dir) / audio_path
        mel = audioprocessor.file2spectrogram(
            audio_path, return_wav=False, is_trim=False
        )
        mel = torch.from_numpy(mel).float().to(device)
        mels.append(mel)
    print(f"[INFO]: {len(mels)} audios is loaded.")

    with torch.no_grad():
        waveforms = batch_inference(mels, vocoder, batch_size=batch_size)

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, waveform in enumerate(waveforms):
        output_path = save_dir / f"{i:04d}.wav"
        metadata["pairs"][i]["converted"] = f"{i:04d}.wav"
        waveform = waveform.detach().cpu().numpy()
        sf.write(output_path, waveform, vocoder.sample_rate)

    with open(save_dir / "metadata.json", "w") as file_pointer:
        json.dump(metadata, file_pointer, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
