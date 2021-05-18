"""Inferencer of Stargan-vc"""
from typing import List
from pathlib import Path
import json
import numpy as np
import torch
from torch import Tensor

from .model import GeneratorSplit, SPEncoder
from .audioprocessor import AudioProcessor
from .utils import *


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        status_path = checkpoint_dir / "status"
        speaker_used_path = checkpoint_dir / "speaker_used.json"
        checkpoint_path = checkpoint_dir / "model.ckpt"
        spk_checkpoint_path = checkpoint_dir / "speaker_encoder.ckpt"

        speakers = json.load(speaker_used_path.open())

        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        spk_checkpoint = torch.load(
            spk_checkpoint_path, map_location=lambda storage, loc: storage)

        self.model = GeneratorSplit(num_speakers=len(
            speakers), aff=True, res_block_name="Style2ResidualBlock1DBeta").to(device)
        self.model.load_state_dict(checkpoint)

        self.speaker_encoder = SPEncoder(
            num_speakers=len(speakers), spk_cls=False).to(device)
        self.speaker_encoder.load_state_dict(spk_checkpoint)

        self.speakers = speakers
        self.status_path = status_path
        self.device = device
        self.sample_rate = AudioProcessor.sample_rate
        self.frame_period = AudioProcessor.frame_period

    def get_speaker_id(self, speaker_name: str):
        speaker_id = self.speakers.index(speaker_name)
        # if isinstance(speaker_name, str):
        #     try:
        #         speaker_id = self.speakers[speaker_name]
        #     except ValueError:
        #         print(f"{speaker_name} is not available")
        #         exit()

        return torch.LongTensor([int(speaker_id)]).to(self.device)

    def get_speaker_conds(self, speaker_name, feat, status):
        speaker_id = self.get_speaker_id(speaker_name)
        _, _, _, _, coded_sp = feat
        coded_sp = (coded_sp - status["coded_sps_mean"]
                    ) / status["coded_sps_std"]
        coded_sp = torch.FloatTensor(coded_sp.T).unsqueeze(
            0).unsqueeze(1).to(self.device)
        speaker_conds = self.speaker_encoder(coded_sp, speaker_id)

        return speaker_conds

    def inference(self, src_speaker: str, tgt_speaker: str, src_feat, tgt_feat) -> Tensor:
        """Inference one utterance."""
        with torch.no_grad():
            src_status = np.load(
                self.status_path / f"{src_speaker}_stats.npz")
            tgt_status = np.load(
                self.status_path / f"{tgt_speaker}_stats.npz")

            src_conds = self.get_speaker_conds(
                src_speaker, src_feat, src_status)
            tgt_conds = self.get_speaker_conds(
                tgt_speaker, tgt_feat, tgt_status)

            f0, _, _, ap, coded_sp = src_feat
            coded_sp = (
                coded_sp - src_status["coded_sps_mean"]) / src_status["coded_sps_std"]
            coded_sp = torch.FloatTensor(coded_sp.T).unsqueeze(
                0).unsqueeze(1).to(self.device)

            f0_converted = pitch_conversion(
                f0=f0,
                mean_log_src=src_status["log_f0s_mean"],
                std_log_src=src_status["log_f0s_std"],
                mean_log_target=tgt_status["log_f0s_mean"],
                std_log_target=tgt_status["log_f0s_std"]
            )

            coded_sp_converted = self.model(
                coded_sp, src_conds, tgt_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(
                coded_sp_converted).T * tgt_status["coded_sps_std"] + tgt_status["coded_sps_mean"]
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            output = world_speech_synthesis(
                f0=f0_converted, coded_sp=coded_sp_converted, ap=ap, fs=self.sample_rate, frame_period=self.frame_period)

        return torch.from_numpy(output)

    def inference_from_path(self, source_speaker: str, target_speaker: str, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        try:
            src_feat = AudioProcessor.file2spectrogram(src_path)
        except ValueError:
            src_feat = AudioProcessor.file2spectrogram(src_path, is_trim=False)

        try:
            tgt_feat = AudioProcessor.file2spectrogram(tgt_paths[0])
        except ValueError:
            tgt_feat = AudioProcessor.file2spectrogram(
                tgt_paths[0], is_trim=False)

        output = self.inference(
            source_speaker, target_speaker, src_feat, tgt_feat)

        return output

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_speaker = pair["source_speaker"]
        target_speaker = pair["target_speaker"]

        source_utt = Path(source_dir) / pair["src_utt"]
        target_utt = Path(source_dir) / pair["tgt_utts"][0]
        output = self.inference_from_path(
            source_speaker, target_speaker, source_utt, [target_utt])

        return output

    def spectrogram2waveform(self, waveforms: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""

        return waveforms
