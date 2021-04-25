#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import numpy as np
from librosa import load, stft
from librosa.filters import mel
from librosa.effects import trim


class AudioProcessor:
    """Process audio data."""

    # hyperparameters
    sample_rate = 16000
    hop_len = 256
    fft_len = 1024
    n_mels = 80

    @classmethod
    def load_wav(cls, file_path):
        """Load waveform."""

    @classmethod
    def wav2spectrogram(cls, wav):
        """Waveform to spectrogram."""

    @classmethod
    def file2spectrogram(cls, file_path, return_wav=False):
        """Load audio file and create spectrogram."""
        wav = cls.load_wav(file_path)
        spectrogram = cls.wav2spectrogram(wav)

        if return_wav:
            return wav, spectrogram

        return spectrogram

    @classmethod
    def get_input(cls, file_path):
        """Get model input."""

        wav = cls.load_wav(file_path)
        d_mel = cls.wav2spectrogram(wav)

        return d_mel
