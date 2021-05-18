"""Inference one utterance."""
import warnings
from datetime import datetime
import importlib
from argparse import ArgumentParser
from pathlib import Path

import torch


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", type=str, help="The directrory of test data.", required=True
    )
    parser.add_argument(
        "-r", "--root", type=str, help="The objective metric directrory.", required=True
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None, help="The output path."
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        default=None,
        help="The directrory of label data.",
    )
    parser.add_argument(
        "-th",
        "--threshold_path",
        type=str,
        default=None,
        help="The path of threshold.",
    )
    parser.add_argument(
        "-m",
        "--metadata_path",
        type=str,
        default=None,
        help="The path of metadata.",
    )
    parser.add_argument(
        "-l", "--language", type=str, default="EN", help="The language for ASR."
    )

    return vars(parser.parse_args())


def main(data_dir, output_dir, root, target_dir, threshold_path, metadata_path, language):
    """Main function"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import Inferencer module
    inference_path = Path(root) / "inference"
    inference_path = str(inference_path).replace("/", ".")
    load_model = getattr(importlib.import_module(inference_path), "load_model")
    calculate_score = getattr(
        importlib.import_module(inference_path), "calculate_score"
    )

    print(f"[INFO]: The metric is used from {root}.")
    if root.find("character_error_rate") > -1:
        root = language
    model = load_model(root, device)

    with torch.no_grad():
        print(f"[INFO]: The testing waveform is loaded from {data_dir}.")
        step_moment = datetime.now()

        arguments = {
            "model": model,
            "device": device,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "metadata_path": metadata_path,
            "target_dir": target_dir,
            "threshold_path": threshold_path,
        }

        calculate_score(**arguments)
        elaspe_time = datetime.now() - step_moment
        print("[INFO]: The time of calculate score",
              elaspe_time.total_seconds())
        print("-" * 100)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
