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
        "-d", "--data_dir", type=str, help="The directrory of test data."
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        default=None,
        help="The directrory of label data.",
    )
    parser.add_argument(
        "--threshold_path",
        type=str,
        default=None,
        help="The path of threshold.",
    )
    parser.add_argument("-o", "--output", type=str, help="The output path.")
    parser.add_argument("-r", "--root", type=str, help="The model directrory.")

    return vars(parser.parse_args())


def main(data_dir, output, root, threshold_path, target_dir):
    """Main function"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import Inferencer module
    inference_path = Path(root) / "inference"
    inference_path = str(inference_path).replace("/", ".")
    load_model = getattr(importlib.import_module(inference_path), "load_model")
    calculate_score = getattr(
        importlib.import_module(inference_path), "calculate_score"
    )

    model = load_model(root)
    print(f"[INFO]: The metric is used from {root}.")

    with torch.no_grad():
        print(f"[INFO]: The testing waveform is loaded from {data_dir}.")
        step_moment = datetime.now()

        arguments = {
            "model": model,
            "device": device,
            "data_dir": data_dir,
            "target_dir": target_dir,
            "output": output,
            "threshold_path": threshold_path,
        }

        calculate_score(**arguments)
        elaspe_time = datetime.now() - step_moment
        print("[INFO]: The time of calculate score", elaspe_time.total_seconds())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
