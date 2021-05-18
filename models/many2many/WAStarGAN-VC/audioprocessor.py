#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

from typing import Tuple, Optional
import numpy as np
from librosa import load, stft
from librosa.effects import trim

from .utils import *


class AudioProcessor:
    """Process audio data."""

    # Sample rate.
    sample_rate = 22050
    num_mcep = 36
    frame_period = 5

    @classmethod
    def load_wav(cls, file_path: str, is_trim: bool) -> np.ndarray:
        """Load waveform."""
        wav, _ = load(file_path, sr=cls.sample_rate, mono=True)
        # Trimming
        if is_trim:
            wav, _ = trim(wav)

        return wav

    @classmethod
    def wav2spectrogram(cls, wav: np.ndarray) -> np.ndarray:
        """Waveform to spectrogram."""
        f0, timeaxis, sp, ap = world_decompose(
            wav=wav, fs=cls.sample_rate, frame_period=cls.frame_period)
        coded_sp = world_encode_spectral_envelop(
            sp=sp, fs=cls.sample_rate, dim=cls.num_mcep)

        return (f0, timeaxis, sp, ap, coded_sp)

    @classmethod
    def file2spectrogram(
        cls, file_path, return_wav=False, is_trim=False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load audio file and create spectrogram."""
        wav = cls.load_wav(file_path, is_trim=is_trim)
        spectrogram = cls.wav2spectrogram(wav)

        if return_wav:
            return wav, spectrogram

        return spectrogram
