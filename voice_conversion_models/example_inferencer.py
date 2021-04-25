"""Inferencer of AdaIN-VC"""
from pathlib import Path
import numpy as np
import torch


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        self.root = Path(root)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def inference(self, src_mel, tgt_mels):
        """Inference one utterance."""

        return conv_mel
