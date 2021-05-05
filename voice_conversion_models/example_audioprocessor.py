#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

from typing import Tuple, Optional
import numpy as np


class AudioProcessor:
    """Process audio data."""

    # hyperparameters
    sample_rate = 16000
    hop_len = 256
    fft_len = 1024
    n_mels = 80

    @classmethod
    def load_wav(cls, file_path: str, is_trim: bool) -> np.ndarray:
        """Load waveform."""

    @classmethod
    def wav2spectrogram(cls, wav: np.ndarray) -> np.ndarray:
        """Waveform to spectrogram."""

    @classmethod
    def file2spectrogram(
        cls, file_path, return_wav=False, is_trim=True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load audio file and create spectrogram."""

        wav = cls.load_wav(file_path, is_trim=is_trim)
        d_mel = cls.wav2spectrogram(wav)

        if return_wav:
            return wav, d_mel

        return d_mel
