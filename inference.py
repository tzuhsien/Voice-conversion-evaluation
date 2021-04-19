"""Inference one utterance."""
import warnings
from datetime import datetime
import importlib
from argparse import ArgumentParser
from pathlib import Path
import soundfile as sf

import torch


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="source wav path")
    parser.add_argument("-t", "--targets", nargs="+", help="target wav path")
    parser.add_argument("-o", "--output", type=str, help="output wav path")
    parser.add_argument("-r", "--root", type=str, help="the model dir")

    return vars(parser.parse_args())


def main(source, targets, output, root):
    """Main function"""

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

    with torch.no_grad():
        print(f"[INFO]: Source waveform is loaded from {source}.")
        print(f"[INFO]: Target waveforms is loaded from {targets}.")
        step_moment = datetime.now()

        # conv_result: Tensor at cpu with shape (length, n_mels)
        conv_result = inferencer.inference_from_file(source, targets)
        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO]: The time of converting audio", elaspe_time.total_seconds())

        conv_result = conv_result.to(device)
        waveform = vocoder.generate([conv_result])[0]
        elaspe_time = datetime.now() - step_moment
        print("[INFO]: The time of generating waveform", elaspe_time.total_seconds())

    waveform = waveform.detach().cpu().numpy()
    sf.write(output, waveform, vocoder.sample_rate)
    print(f"[INFO]: Save converted waveforms to {output}.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
