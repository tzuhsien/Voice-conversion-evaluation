"""Inferencer of AdaIN-VC"""
from typing import List
from pathlib import Path
import yaml
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import AE
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = checkpoint_dir / "config.yaml"
        model_path = checkpoint_dir / "model.ckpt"
        attr_path = checkpoint_dir / "attr.pkl"
        vocoder_path = checkpoint_dir / "vocoder.pt"
        config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

        self.model = AE(config).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.vocoder = torch.jit.load(str(vocoder_path)).to(device)

        self.attr = joblib.load(attr_path)

        self.device = device
        self.frame_size = config["data_loader"]["frame_size"]
        self.sample_rate = AudioProcessor.sample_rate

    def denormalize(self, d_mel: np.ndarray) -> np.ndarray:
        """denormalize"""
        mean, std = self.attr["mean"], self.attr["std"]
        ret = d_mel * std + mean

        return ret

    def normalize(self, d_mel: np.ndarray) -> np.ndarray:
        """normalize"""
        mean, std = self.attr["mean"], self.attr["std"]
        ret = (d_mel - mean) / std

        return ret

    def preprocess(self, d_mel: np.ndarray) -> np.ndarray:
        """Preprocess"""
        d_mel = torch.from_numpy(self.normalize(d_mel)).to(self.device)

        remains = d_mel.size(0) % self.frame_size
        if remains != 0:
            d_mel = F.pad(d_mel, (0, remains))

        d_mel = d_mel.view(
            -1, d_mel.size(0) // self.frame_size, self.frame_size * d_mel.size(1)
        ).transpose(1, 2)

        return d_mel

    def inference(self, src_mel: np.ndarray, tgt_mels: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""
        src_mel = self.preprocess(src_mel)
        tgt_mels = [self.preprocess(tgt_mel) for tgt_mel in tgt_mels]

        with torch.no_grad():
            latent = self.model.encode_linguistic(src_mel)
            tgt_embs = [self.model.encode_timbre(tgt_mel) for tgt_mel in tgt_mels]
            tgt_emb = torch.mean(torch.cat(tgt_embs), 0, keepdim=True)
            dec = self.model.decode(latent, tgt_emb)
            dec = dec.transpose(1, 2).squeeze(0)
            dec = dec.detach().cpu().numpy()
            dec = self.denormalize(dec)

        return torch.from_numpy(dec)

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
