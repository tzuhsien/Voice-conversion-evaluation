"""Inferencer of FragmentVC"""
from typing import List
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from .model import VC_MODEL
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(root) / "checkpoints"
        ckpt_path = checkpoint_dir / "vqvc+.pt"
        vocoder_path = checkpoint_dir / "vocoder.pt"

        self.model = VC_MODEL().eval().to(device)
        self.model.load_state_dict(torch.load(ckpt_path))

        self.vocoder = torch.jit.load(str(vocoder_path)).eval().to(device)
        self.device = device

        self.sample_rate = AudioProcessor.sample_rate

    def inference(self, src_mel: np.ndarray, tgt_mels: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""
        src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(self.device)
        src_mel = src_mel[:, :, : src_mel.size(2) // 16 * 16]

        tgt_mels = [
            torch.from_numpy(tgt_mel).T.unsqueeze(0).to(self.device)
            for tgt_mel in tgt_mels
        ]
        tgt_mel = torch.cat(tgt_mels, 2)
        tgt_mel = tgt_mel[:, :, : tgt_mel.size(2) // 16 * 16]

        with torch.no_grad():
            out_mel = self.model.inference(src_mel, tgt_mel)

        return out_mel.to("cpu")

    def inference_from_path(self, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        try:
            src_mel = AudioProcessor.file2spectrogram(src_path)
        except ValueError:
            src_mel = AudioProcessor.file2spectrogram(src_path, is_trim=True)

        tgt_mels = []
        for tgt_path in tgt_paths:
            try:
                tgt_mel = AudioProcessor.file2spectrogram(tgt_path)
            except ValueError:
                tgt_mel = AudioProcessor.file2spectrogram(tgt_path, is_trim=False)

            tgt_mels.append(tgt_mel)

        result = self.inference(src_mel, tgt_mels)

        return result.squeeze(0).T.to("cpu")

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        target_utts = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
        conv_mel = self.inference_from_path(source_utt, target_utts)

        return conv_mel

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        waveforms = self.vocoder.generate(spectrogram)

        return waveforms
