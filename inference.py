"""Inference one utterance."""
import warnings
from datetime import datetime
import importlib
from argparse import ArgumentParser
from pathlib import Path
import soundfile as sf
import logging

import torch


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)-s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


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
    logger.info("Inferencer is loaded from %s.", root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder_path = Path(root) / "checkpoints/vocoder.pt"
    vocoder = torch.jit.load(str(vocoder_path)).to(device)
    logger.info("Vocoder is loaded from %s.", vocoder_path)

    with torch.no_grad():
        logger.info("Source waveform is loaded from %s.", source)
        logger.info("Target waveforms is loaded from %s.", targets)
        step_moment = datetime.now()

        # conv_result: Tensor at cpu with shape (length, n_mels)
        conv_result = inferencer.inference_from_path(source, targets)
        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        logger.info("The time of converting audio: %.1f s", elaspe_time.total_seconds())

        conv_result = conv_result.to(device)
        waveform = vocoder.generate([conv_result])[0]
        elaspe_time = datetime.now() - step_moment
        logger.info("The time of generating waveform: %.1f s", elaspe_time.total_seconds())

    waveform = waveform.detach().cpu().numpy()
    sf.write(output, waveform, vocoder.sample_rate)
    logger.info("Save converted waveforms to %s.", output)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
