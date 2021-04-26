"""Inferencer of AdaIN-VC"""
from typing import List
from pathlib import Path
import numpy as np
import torch

from .models import load_pretrained_wav2vec
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        root = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wav2vec_path = str(root / "wav2vec_small.pt")
        ckpt_path = str(root / "fragmentvc.pt")

        self.wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
        self.model = torch.jit.load(ckpt_path).to(device).eval()
        self.device = device

    def inference_one_utterance(self, src_wav, tgt_mels):
        """Inference one utterance."""
        src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(self.device)
        tgt_mel = np.concatenate(tgt_mels, axis=0)
        tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(self.device)
        with torch.no_grad():
            src_feat = self.wav2vec.extract_features(src_wav, None)[0]

            out_mel, _ = self.model(src_feat, tgt_mel)
            out_mel = out_mel.transpose(1, 2).squeeze(0)

        return out_mel

    def inference_from_file(self, src_path: str, tgt_paths: List):
        """Inference from path."""
        src_wav, _ = AudioProcessor.file2spectrogram(src_path, return_wav=True)
        tgt_mels = [AudioProcessor.file2spectrogram(tgt_path) for tgt_path in tgt_paths]

        conv_mel = self.inference_one_utterance(src_wav, tgt_mels)

        return conv_mel
