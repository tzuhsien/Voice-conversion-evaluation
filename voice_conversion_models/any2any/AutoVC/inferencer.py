"""Inferencer of AdaIN-VC"""
from typing import List
from collections import OrderedDict

from pathlib import Path
from math import ceil
import numpy as np
import torch
from torch import Tensor


from .model_bl import D_VECTOR
from .model_vc import Generator
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_dir = Path(root) / "checkpoints"
        generator_path = checkpoint_dir / "autovc.ckpt"
        dvector_path = checkpoint_dir / "encoder.ckpt"
        vocoder_path = checkpoint_dir / "vocoder.pt"

        dvector_ckpt = torch.load(dvector_path, map_location=device)
        state_dict = OrderedDict()
        for key, val in dvector_ckpt["model_b"].items():
            state_dict[key[7:]] = val

        generator_ckpt = torch.load(generator_path, map_location=device)

        self.dvector = (
            D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
        )
        self.dvector.load_state_dict(state_dict)

        self.generator = Generator(32, 256, 512, 32).eval().to(device)
        self.generator.load_state_dict(generator_ckpt["model"])

        self.vocoder = torch.jit.load(str(vocoder_path)).to(device)

        self.device = device
        self.sample_rate = AudioProcessor.sample_rate

    def encode_uttr(self, mel: np.ndarray) -> Tensor:
        """Encode timbre information."""
        melsp = torch.from_numpy(mel[np.newaxis, :, :]).to(self.device)
        emb = self.dvector(melsp)
        return emb

    @classmethod
    def pad_seq(cls, mel: np.ndarray, base=32) -> np.ndarray:
        """Pad melspectrogram."""
        len_out = int(base * ceil(float(mel.shape[0]) / base))
        len_pad = len_out - mel.shape[0]
        assert len_pad >= 0
        return np.pad(mel, ((0, len_pad), (0, 0)), "constant"), len_pad

    def conversion(self, src_emb: Tensor, tgt_emb: Tensor, uttr: np.ndarray) -> Tensor:
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

    def inference(self, src_mel: np.ndarray, tgt_mels: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""
        with torch.no_grad():
            src_emb = self.encode_uttr(src_mel).squeeze()
            tgt_embs = [self.encode_uttr(tgt_mel).squeeze() for tgt_mel in tgt_mels]
            tgt_emb = torch.stack(tgt_embs).mean(dim=0)

            result = self.conversion(src_emb, tgt_emb, src_mel)

        return result

    def inference_from_path(self, source: Path, targets: List[Path]) -> Tensor:
        """Inference from file path."""
        src_mel = AudioProcessor.file2spectrogram(source)
        tgt_mels = [AudioProcessor.file2spectrogram(target) for target in targets]

        result = self.inference(src_mel, tgt_mels)

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
