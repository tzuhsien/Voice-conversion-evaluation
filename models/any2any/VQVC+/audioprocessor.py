#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

from typing import Tuple, Optional
import numpy as np
from librosa import load, stft
from librosa.filters import mel
from librosa.effects import trim


class AudioProcessor:
    """Process audio data."""

    # hyperparameters
    sample_rate = 22050
    n_fft = 1024
    win_len = 1024
    hop_len = 256
    n_mels = 80

    f_min = 0
    f_max = 11025

    top_db = 30

    mel_basis = mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )

    @classmethod
    def load_wav(cls, file_path: str, is_trim: bool) -> np.ndarray:
        """Load waveform."""
        wav = load(file_path, sr=cls.sample_rate)[0]
        if is_trim:
            wav = trim(wav, top_db=cls.top_db)[0]
        wav = np.clip(wav, -1.0 + 1e-6, 1.0 - 1e-6)

        return wav

    @classmethod
    def wav2spectrogram(cls, wav: np.ndarray) -> np.ndarray:
        """Waveform to spectrogram."""

        magnitude = np.abs(
            stft(wav, n_fft=cls.n_fft, hop_length=cls.hop_len, win_length=cls.win_len)
        )
        mel_spec = np.dot(cls.mel_basis, magnitude)
        log_mel_spec = np.log10(mel_spec + 1e-9)
        return log_mel_spec.T

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
