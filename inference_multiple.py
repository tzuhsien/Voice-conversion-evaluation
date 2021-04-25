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
    parser.add_argument("--reload_dir", type=str, help="output wav path")
    parser.add_argument("--no_vocoder", action="store_true")

    return vars(parser.parse_args())


def batch_inference(out_mels, vocoder, batch_size):
    out_wavs = []
    for i in tqdm(range(0, len(out_mels), batch_size)):
        right = i + batch_size if len(out_mels) - i >= batch_size else len(out_mels)
        out_wavs.extend(vocoder.generate(out_mels[i:right]))
    return out_wavs


def main(
    metadata_path,
    source_dir,
    target_dir,
    output_dir,
    root,
    batch_size,
    reload,
    reload_dir,
    no_vocoder,
):
    """Main function"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)

    # import Inferencer module
    inferencer_path = Path(root) / "inferencer"
    inferencer_path = str(inferencer_path).replace("/", ".")
    Inferencer = getattr(importlib.import_module(inferencer_path), "Inferencer")

    inferencer = Inferencer(root)
    print(f"[INFO]: Inferencer is loaded from {root}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder_path = Path(root) / "checkpoints/vocoder.pt"
    vocoder = torch.jit.load(str(vocoder_path)).to(device)
    print(f"[INFO]: Vocoder is loaded from {str(vocoder_path)}.")

    metadata = json.load(open(metadata_path))
    print(f"[INFO]: Metadata list is loaded from {metadata_path}.")

    metadata["vc_model"] = root

    conv_mels = []
    if not reload:
        mel_output_dir = output_dir / "mel_files"
        mel_output_dir.mkdir(parents=True)

        for pair in tqdm(metadata["pairs"]):
            # conv_mel: Tensor at cpu with shape ()
            source = Path(source_dir) / pair["src_utt"]
            targets = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
            conv_mel = inferencer.inference_from_path(source, targets)
            conv_mel = conv_mel.detach()

            prefix = Path(pair["src_utt"]).stem
            postfix = Path(pair["tgt_utts"][0]).stem
            file_path = mel_output_dir / f"{prefix}_to_{postfix}.pt"
            torch.save(conv_mel, file_path)

            pair["mel_path"] = f"mel_files/{prefix}_to_{postfix}.pt"
            conv_mels.append(conv_mel.to(device))

        metadata_output_path = output_dir / "metadata.json"
        json.dump(metadata, metadata_output_path.open())

    else:
        for pair in tqdm(metadata["pairs"]):
            file_path = Path(reload_dir) / pair["mel_path"]
            conv_mel = torch.load(file_path)
            conv_mels.append(conv_mel.to(device))

    del inferencer
    if no_vocoder:
        return

    with torch.no_grad():
        waveforms = batch_inference(conv_mels, vocoder, batch_size=batch_size)

    for pair, waveform in tqdm(zip(metadata["pairs"], waveforms)):
        waveform = waveform.detach().cpu().numpy()

        prefix = Path(pair["src_utt"]).stem
        postfix = Path(pair["tgt_utts"][0]).stem
        file_path = output_dir / f"{prefix}_to_{postfix}.wav"
        pair["converted"] = f"{prefix}_to_{postfix}.wav"

        sf.write(file_path, waveform, vocoder.sample_rate)

    metadata["vocoder"] = str(vocoder_path)
    metadata_output_path = output_dir / "metadata.json"
    json.dump(metadata, metadata_output_path.open())


if __name__ == "__main__":
    main(**parse_args())
