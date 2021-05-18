#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

from typing import Tuple, Optional
import numpy as np
from librosa import load, stft
from librosa.effects import trim


class AudioProcessor:
    """Process audio data."""

    # signal processing
    sample_rate = 16000  # Sample rate.
    n_fft = 1024  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_len = int(sample_rate * frame_shift)  # samples.
    win_len = int(sample_rate * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    preemphasis = 0.97  # or None
    max_db = 100
    ref_db = 20

    @classmethod
    def load_wav(cls, file_path: str, is_trim: bool) -> np.ndarray:
        """Load waveform."""
        wav, _ = load(file_path, sr=cls.sample_rate)
        # Trimming
        if is_trim:
            wav, _ = trim(wav)

        wav = wav / (np.abs(wav).max() + 1e-6)

        return wav

    @classmethod
    def wav2spectrogram(cls, wav: np.ndarray) -> np.ndarray:
        """Waveform to spectrogram."""
        # Preemphasis
        wav = np.append(wav[0], wav[1:] - cls.preemphasis * wav[:-1])
        # stft
        linear = stft(
            y=wav, n_fft=cls.n_fft, hop_length=cls.hop_len, win_length=cls.win_len
        )

        # magnitude spectrogram
        spec = np.abs(linear)  # (1+n_fft//2, T)
        # to decibel
        spec = 20 * np.log10(np.maximum(1e-5, spec))
        # normalize
        spec = np.clip((spec - cls.ref_db + cls.max_db) / cls.max_db, 1e-8, 1)
        # Transpose
        spec = spec.T.astype(np.float32)  # (T, 1+n_fft//2)

        return spec

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
