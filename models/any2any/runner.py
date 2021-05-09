"""Inferencer of FragmentVC"""
from typing import List
from pathlib import Path
import importlib
from torch import Tensor


class Runner:
    """Inferencer"""

    def __init__(self, root, model_name):
        root = Path(root) / model_name
        inferencer_path = str(root / "inferencer").replace("/", ".")
        Inferencer = getattr(importlib.import_module(inferencer_path), "Inferencer")

        audioprocessor_path = str(root / "audioprocessor").replace("/", ".")
        self.audioprocessor = getattr(
            importlib.import_module(audioprocessor_path), "AudioProcessor"
        )

        self.inferencer = Inferencer(root)

    def inference_from_path(self, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        try:
            src_wav, _ = self.audioprocessor.file2spectrogram(src_path, return_wav=True)
        except ValueError:
            src_wav, _ = self.audioprocessor.file2spectrogram(
                src_path, return_wav=True, is_trim=False
            )

        tgt_mels = []
        for tgt_path in tgt_paths:
            try:
                tgt_mel = self.audioprocessor.file2spectrogram(tgt_path)
            except ValueError:
                tgt_mel = self.audioprocessor.file2spectrogram(tgt_path, is_trim=False)

            tgt_mels.append(tgt_mel)

        result = self.inferencer.inference(src_wav, tgt_mels)

        return result

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        target_utts = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
        conv_mel = self.inference_from_path(source_utt, target_utts)

        return conv_mel

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        waveforms = self.inferencer.spectrogram2waveform(spectrogram)

        return waveforms
