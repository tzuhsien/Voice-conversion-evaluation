"""Inferencer of FragmentVC"""
from typing import List, Optional
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from .models import load_pretrained_wav2vec
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(root) / "checkpoints"
        wav2vec_path = checkpoint_dir / "wav2vec_small.pt"
        ckpt_path = checkpoint_dir / "fragmentvc.pt"
        vocoder_path = checkpoint_dir / "vocoder.pt"

        self.wav2vec = load_pretrained_wav2vec(str(wav2vec_path)).to(device)
        self.model = torch.jit.load(str(ckpt_path)).eval().to(device)
        self.vocoder = torch.jit.load(str(vocoder_path)).eval().to(device)
        self.device = device
        self.sample_rate = AudioProcessor.sample_rate

    def inference(self, src_wav: np.ndarray, tgt_mels: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""
        src_wav = torch.from_numpy(src_wav).unsqueeze(0).to(self.device)
        tgt_mel = np.concatenate(tgt_mels, axis=0)
        tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(self.device)
        with torch.no_grad():
            src_feat = self.wav2vec.extract_features(src_wav, None)[0]

            out_mel, _ = self.model(src_feat, tgt_mel)
            out_mel = out_mel.transpose(1, 2).squeeze(0)

        return out_mel.to("cpu")

    def inference_from_path(self, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        try:
            src_wav, _ = AudioProcessor.file2spectrogram(src_path, return_wav=True)
        except ValueError:
            src_wav, _ = AudioProcessor.file2spectrogram(
                src_path, return_wav=True, is_trim=False
            )

        tgt_mels = []
        for tgt_path in tgt_paths:
            try:
                tgt_mel = AudioProcessor.file2spectrogram(tgt_path)
            except ValueError:
                tgt_mel = AudioProcessor.file2spectrogram(tgt_path, is_trim=False)

            tgt_mels.append(tgt_mel)

        result = self.inference(src_wav, tgt_mels)

        return result

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
