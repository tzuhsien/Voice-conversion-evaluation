"""Inferencer of AdaIN-VC"""
from typing import List
from collections import OrderedDict

import os
from pathlib import Path
from math import ceil
import numpy as np
import torch

from models import load_pretrained_wav2vec
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wav2vec_path = self.root / "checkpoints/encoder.ckpt"
        self.wav2vec = load_pretrained_wav2vec(wav2vec_path).to(self.device)

        ckpt_path = self.root / "checkpoints/encoder.ckpt"
        self.model = torch.jit.load(ckpt_path).to(self.device).eval()

    def conversion(self, src_emb, tgt_emb, uttr):
        """Convert timbre from source to target."""
        x_org, len_pad = self.pad_seq(uttr)
        uttr = torch.from_numpy(x_org[np.newaxis, :, :]).to(self.device)
        src_emb = src_emb.unsqueeze(0)
        tgt_emb = tgt_emb.unsqueeze(0)
        _, x_identic_psnt, _ = self.generator(uttr, src_emb, tgt_emb)
        if len_pad == 0:
            result = x_identic_psnt[0, 0, :, :].cpu()
        else:
            result = x_identic_psnt[0, 0, :-len_pad, :].cpu()

        return result

    def get_spectrogram(self, path: str):
        """Extract melsoectrogram from path."""
        d_mel = AudioProcessor.file2spectrogram(path)

        return d_mel

    def inference_one_utterance(self, src_mel, tgt_mels):
        """Inference one utterance."""
        with torch.no_grad():
            src_emb = self.encode_uttr(src_mel).squeeze()
            tgt_embs = [self.encode_uttr(tgt_mel).squeeze() for tgt_mel in tgt_mels]
            tgt_emb = torch.stack(tgt_embs).mean(dim=0)

            result_mel = self.conversion(src_emb, tgt_emb, src_mel)

        return result_mel

    def inference_from_path(self, src_path: str, tgt_paths: List):
        """Inference from path."""
        src_mel = self.get_spectrogram(src_path)
        tgt_mels = [self.get_spectrogram(tgt_path) for tgt_path in tgt_paths]

        conv_mel = self.inference_one_utterance(src_mel, tgt_mels)

        return conv_mel
