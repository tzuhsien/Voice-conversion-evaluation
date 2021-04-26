#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import numpy as np
from librosa import load, stft
from librosa.filters import mel
from librosa.effects import trim
from scipy.signal import butter, filtfilt


class AudioProcessor:
    """Process audio data."""

    sample_rate = 16000
    n_mels = 80
    top_db = 60
    ref_db = 20
    max_db = 100
    fft_len = 1024
    hop_len = 256
    fmin = 90
    fmax = 7600
    mel_basis = mel(sample_rate, fft_len, fmin=fmin, fmax=fmax, n_mels=n_mels).T
    min_level = np.exp(-100 / 20 * np.log(10))

    @classmethod
    def load_wav(cls, file_path):
        """Load waveform."""
        wav = load(file_path, sr=cls.sample_rate)[0]
        wav = wav / (np.abs(wav).max() + 1e-6)
        wav = trim(wav, top_db=cls.top_db)[0]
        wav = filtfilt(*cls.butter_highpass(), wav)
        wav = wav * 0.96

        return wav

    @classmethod
    def wav2spectrogram(cls, wav):
        """Waveform to spectrogram."""
        d_mag = cls.short_time_fourier_transform(wav)
        d_mel = np.dot(d_mag.T, cls.mel_basis)

        db_val = 20 * np.log10(np.maximum(cls.min_level, d_mel))
        db_scaled = db_val - cls.ref_db
        db_normalized = (db_scaled + cls.max_db) / cls.max_db
        db_normalized = np.clip(db_normalized, 0, 1).astype(np.float32)

        return db_normalized

    @classmethod
    def file2spectrogram(cls, file_path, return_wav=False):
        """Load audio file and create spectrogram."""
        wav = cls.load_wav(file_path)

        spectrogram = cls.wav2spectrogram(wav)

        if return_wav:
            return wav, spectrogram

        return spectrogram

    @classmethod
    def butter_highpass(cls, cutoff=30, order=5):
        """Create butter highpass filter."""

        normal_cutoff = cutoff / (0.5 * cls.sample_rate)
        return butter(order, normal_cutoff, btype="high", analog=False)

    @classmethod
    def short_time_fourier_transform(cls, wav):
        """Apply short time Fourier transform."""

        d_matrix = stft(wav, n_fft=cls.fft_len, hop_length=cls.hop_len)
        return np.abs(d_matrix)
