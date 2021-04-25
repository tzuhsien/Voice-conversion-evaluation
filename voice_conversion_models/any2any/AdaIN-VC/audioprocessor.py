#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import numpy as np
from librosa import load, stft
from librosa.filters import mel
from librosa.effects import trim


class AudioProcessor:
    """Process audio data."""

    # signal processing
    sample_rate = 24000  # Sample rate.
    fft_len = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_len = int(sample_rate * frame_shift)  # samples.
    win_len = int(sample_rate * frame_length)  # samples.
    n_mels = 512  # Number of Mel banks to generate
    preemphasis = 0.97  # or None
    max_db = 100
    ref_db = 20
    top_db = 15

    @classmethod
    def load_wav(cls, file_path):
        """Load waveform."""
        wav, _ = load(file_path, sr=cls.sample_rate)
        wav = wav / (np.abs(wav).max() + 1e-6)
        wav, _ = trim(wav, top_db=cls.top_db)

        return wav

    @classmethod
    def wav2spectrogram(cls, wav):
        """Returns normalized log(melspectrogram).
        Args:
        sound_file: A string. The full path of a sound file.

        Returns:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
        """
        # Preemphasis
        wav = np.append(wav[0], wav[1:] - cls.preemphasis * wav[:-1])

        # stft
        linear = stft(
            y=wav, n_fft=cls.fft_len, hop_length=cls.hop_len, win_length=cls.win_len
        )

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = mel(
            cls.sample_rate, cls.fft_len, cls.n_mels
        )  # (n_mels, 1+n_fft//2)
        d_mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        d_mel = 20 * np.log10(np.maximum(1e-5, d_mel))

        # normalize
        d_mel = np.clip((d_mel - cls.ref_db + cls.max_db) / cls.max_db, 1e-8, 1)

        # Transpose
        d_mel = d_mel.T.astype(np.float32)  # (T, n_mels)

        return d_mel

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
        spectrogram = cls.wav2spectrogram(wav)

        return spectrogram