"""Inferencer of AdaIN-VC"""
from pathlib import Path
import numpy as np
import torch

from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(root) / "checkpoints"

        self.device = device
        self.sample_rate = AudioProcessor.sample_rate

    def inference(self, src_wav: np.ndarray, tgt_mels: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""

        return conv_mel

    def inference_from_path(self, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        try:
            src_wav, _ = AudioProcessor.file2spectrogram(src_path, return_wav=True)
        except ValueError:
            src_wav, _ = AudioProcessor.file2spectrogram(
                src_path, return_wav=True, is_trim=False
            )

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        target_utts = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
        conv_mel = self.inference_from_path(source_utt, target_utts)

        return conv_mel

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""

        return spectrogram
