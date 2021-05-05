"""Inference one utterance."""
import json
from pathlib import Path
import importlib
from argparse import ArgumentParser
from tqdm import tqdm
import soundfile as sf

import torch


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--metadata_path", type=str, help="inference metadata path"
    )
    parser.add_argument("-s", "--source_dir", type=str, help="source dir path")
    parser.add_argument("-t", "--target_dir", type=str, help="target dir path")
    parser.add_argument("-o", "--output_dir", type=str, help="output wav path")
    parser.add_argument("-r", "--root", type=str, help="the model dir")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="the model dir"
    )
    parser.add_argument("--reload", action="store_true")

    return vars(parser.parse_args())


def conversion(
    inferencer, device, root, metadata_path, source_dir, target_dir, output_dir
):
    """Do conversion and save the output of voice conversion model."""
    metadata = json.load(open(metadata_path))
    print(f"[INFO]: Metadata list is loaded from {metadata_path}.")
    metadata["vc_model"] = root
    mel_output_dir = output_dir / "mel_files"
    mel_output_dir.mkdir(parents=True, exist_ok=True)

    conv_mels = []
    for pair in tqdm(metadata["pairs"]):
        # conv_mel: Tensor at cpu with shape ()
        conv_mel = inferencer.inference_from_pair(pair, source_dir, target_dir)
        conv_mel = conv_mel.detach()

        prefix = Path(pair["src_utt"]).stem
        postfix = Path(pair["tgt_utts"][0]).stem
        file_path = mel_output_dir / f"{prefix}_to_{postfix}.pt"
        torch.save(conv_mel, file_path)

        pair["mel_path"] = f"mel_files/{prefix}_to_{postfix}.pt"
        conv_mels.append(conv_mel.to(device))

    metadata_output_path = output_dir / "metadata.json"
    json.dump(metadata, metadata_output_path.open("w"), indent=2)

    return metadata, conv_mels


def reload_from_numpy(device, metadata_path, source_dir):
    """Reload the output of voice conversion model."""
    metadata = json.load(open(metadata_path))
    conv_mels = []
    for pair in tqdm(metadata["pairs"]):
        file_path = Path(source_dir) / pair["mel_path"]
        conv_mel = torch.load(file_path)
        conv_mels.append(conv_mel.to(device))

    return metadata, conv_mels


def main(
    metadata_path,
    source_dir,
    target_dir,
    output_dir,
    root,
    batch_size,
    reload,
):
    """Main function"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # import Inferencer module
    inferencer_path = str(Path(root) / "inferencer").replace("/", ".")
    Inferencer = getattr(importlib.import_module(inferencer_path), "Inferencer")

    inferencer = Inferencer(root)
    device = inferencer.device
    sample_rate = inferencer.sample_rate
    print(f"[INFO]: Inferencer is loaded from {root}.")

    if reload:
        metadata, conv_mels = reload_from_numpy(device, metadata_path, source_dir)
    else:
        metadata, conv_mels = conversion(
            inferencer, device, root, metadata_path, source_dir, target_dir, output_dir
        )

    waveforms = []
    with torch.no_grad():
        number = len(conv_mels)
        for left in tqdm(range(0, number, batch_size)):
            right = left + min(batch_size, number - left)
            waveforms.extend(inferencer.spectrogram2waveform(conv_mels[left:right]))

    for pair, waveform in tqdm(zip(metadata["pairs"], waveforms)):
        waveform = waveform.detach().cpu().numpy()

        prefix = Path(pair["src_utt"]).stem
        postfix = Path(pair["tgt_utts"][0]).stem
        file_path = output_dir / f"{prefix}_to_{postfix}.wav"
        pair["converted"] = f"{prefix}_to_{postfix}.wav"

        sf.write(file_path, waveform, sample_rate)

    metadata_output_path = output_dir / "metadata.json"
    json.dump(metadata, metadata_output_path.open("w"), indent=2)


if __name__ == "__main__":
    main(**parse_args())
