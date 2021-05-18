"""Inferencer of AdaIN-VC"""
from typing import List
from pathlib import Path
import json
import numpy as np
import torch
from torch import Tensor

from .model import Encoder, Decoder
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hyperparameter_path = checkpoint_dir / "hyperparameter.json"
        speaker2id_path = checkpoint_dir / "speaker2id.json"
        checkpoint_path = checkpoint_dir / "model.ckpt"
        vocoder_path = checkpoint_dir / "vocoder.pt"

        hps = json.load(hyperparameter_path.open())
        checkpoint = torch.load(checkpoint_path.open("rb"))

        self.speaker2id = json.load(speaker2id_path.open())

        self.encoder = Encoder(
            ns=hps["ns"], dp=hps["enc_dp"]).to(device).eval()
        self.decoder = (
            Decoder(ns=hps["ns"], c_a=hps["n_speakers"],
                    emb_size=hps["emb_size"])
            .to(device)
            .eval()
        )
        self.generator = (
            Decoder(ns=hps["ns"], c_a=hps["n_speakers"],
                    emb_size=hps["emb_size"])
            .to(device)
            .eval()
        )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.vocoder = torch.jit.load(str(vocoder_path)).to(device)
        self.device = device
        self.sample_rate = AudioProcessor.sample_rate

    def inference(self, src_mel: np.ndarray, target_speaker: str) -> Tensor:
        """Inference one utterance."""
        src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(self.device)
        if isinstance(target_speaker, str):
            try:
                target_speaker = self.speaker2id[target_speaker]
            except ValueError:
                print(f"{target_speaker} is not available")

        target_speaker = torch.LongTensor(
            [int(target_speaker)]).to(self.device)
        with torch.no_grad():
            embedding = self.encoder(src_mel)
            output = self.decoder(embedding, target_speaker)
            output += self.generator(embedding, target_speaker)

        return output.squeeze(0).T.cpu()

    def inference_from_path(self, src_path: Path, target_speaker: str) -> Tensor:
        """Inference from path."""
        try:
            src_mel = AudioProcessor.file2spectrogram(src_path)
        except ValueError:
            src_mel = AudioProcessor.file2spectrogram(src_path, is_trim=False)

        result = self.inference(src_mel, target_speaker)

        return result

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        target_speaker = pair["target_speaker"]
        conv_mel = self.inference_from_path(source_utt, target_speaker)

        return conv_mel

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        waveforms = self.vocoder.generate(spectrogram)

        return waveforms
