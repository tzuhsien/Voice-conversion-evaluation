#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import numpy as np
from librosa import load, stft
from librosa.effects import trim


class AudioProcessor:
    """Process audio data."""

    # signal processing
    sample_rate = 16000  # Sample rate.
    maxmag = 0.99

    @classmethod
    def load_wav(cls, file_path, is_trim):
        """Load waveform."""
        wav, _ = load(file_path, sr=cls.sample_rate)
        # Trimming
        # if is_trim:
        #     wav, _ = trim(wav)

        wav -= np.mean(wav)
        wav *= cls.maxmag/(np.max(np.abs(wav))+1e-7)

        return wav

    @classmethod
    def wav2spectrogram(cls, wav):
        """Waveform to spectrogram."""
        return wav

    @classmethod
    def file2spectrogram(cls, file_path, return_wav=False, is_trim=True):
        """Load audio file and create spectrogram."""
        wav = cls.load_wav(file_path, is_trim=is_trim)
        spectrogram = cls.wav2spectrogram(wav)

        if return_wav:
            return wav, spectrogram

        return spectrogram
