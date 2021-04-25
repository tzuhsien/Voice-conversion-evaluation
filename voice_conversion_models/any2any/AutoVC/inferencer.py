"""Inferencer of AdaIN-VC"""
from collections import OrderedDict

from pathlib import Path
from math import ceil
import numpy as np
import torch

from .model_bl import D_VECTOR
from .model_vc import Generator
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        root = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dvector_path = root / "encoder.ckpt"
        dvector_ckpt = torch.load(dvector_path, map_location=device)
        state_dict = OrderedDict()
        for key, val in dvector_ckpt["model_b"].items():
            state_dict[key[7:]] = val

        generator_path = root / "autovc.ckpt"
        generator_ckpt = torch.load(generator_path, map_location=device)

        self.dvector = (
            D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
        )
        self.dvector.load_state_dict(state_dict)

        # voice conversion

        self.generator = Generator(32, 256, 512, 32).eval().to(device)
        self.generator.load_state_dict(generator_ckpt["model"])

        self.device = device

    def encode_uttr(self, mel):
        """Encode timbre information."""
        melsp = torch.from_numpy(mel[np.newaxis, :, :]).to(self.device)
        emb = self.dvector(melsp)
        return emb

    @classmethod
    def pad_seq(cls, mel, base=32):
        """Pad melspectrogram."""
        len_out = int(base * ceil(float(mel.shape[0]) / base))
        len_pad = len_out - mel.shape[0]
        assert len_pad >= 0
        return np.pad(mel, ((0, len_pad), (0, 0)), "constant"), len_pad

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

    def inference(self, src_mel, tgt_mels):
        """Inference one utterance."""
        with torch.no_grad():
            src_emb = self.encode_uttr(src_mel).squeeze()
            tgt_embs = [self.encode_uttr(tgt_mel).squeeze() for tgt_mel in tgt_mels]
            tgt_emb = torch.stack(tgt_embs).mean(dim=0)

            result = self.conversion(src_emb, tgt_emb, src_mel)

        return result

    def inference_from_file(self, source, targets):
        """Inference from file path."""
        src_mel = AudioProcessor.get_input(source)
        tgt_mels = [AudioProcessor.get_input(target) for target in targets]

        result = self.inference(src_mel, tgt_mels)

        return result
