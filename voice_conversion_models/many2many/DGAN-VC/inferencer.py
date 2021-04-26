"""Inferencer of AdaIN-VC"""
from pathlib import Path
import json
import torch

from .model import Encoder, Decoder
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        root = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hyperparameter_path = root / "hyperparameter.json"
        checkpoint_path = root / "model.ckpt"
        speaker2id_path = root / "speaker2id.json"

        hps = json.load(hyperparameter_path.open())
        checkpoint = torch.load(checkpoint_path.open("rb"))

        self.speaker2id = json.load(speaker2id_path.open())

        self.encoder = Encoder(ns=hps["ns"], dp=hps["enc_dp"]).to(device).eval()
        self.decoder = (
            Decoder(ns=hps["ns"], c_a=hps["n_speakers"], emb_size=hps["emb_size"])
            .to(device)
            .eval()
        )
        self.generator = (
            Decoder(ns=hps["ns"], c_a=hps["n_speakers"], emb_size=hps["emb_size"])
            .to(device)
            .eval()
        )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.device = device

    def inference(self, src_mel, target_speaker):
        """Inference one utterance."""
        src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(self.device)
        if isinstance(target_speaker, str):
            try:
                target_speaker = self.speaker2id[target_speaker]
            except ValueError:
                print(f"{target_speaker} is not available")
        target_speaker = torch.LongTensor([int(target_speaker)]).to(self.device)
        with torch.no_grad():
            embedding = self.encoder(src_mel)
            output = self.decoder(embedding, target_speaker)
            output += self.generator(embedding, target_speaker)

        return output.squeeze(0).T.cpu()

    def inference_from_file(self, source, target_speaker):
        """Inference from file path."""
        src_spec = AudioProcessor.file2spectrogram(source)

        result = self.inference(src_spec, target_speaker[0])

        return result
