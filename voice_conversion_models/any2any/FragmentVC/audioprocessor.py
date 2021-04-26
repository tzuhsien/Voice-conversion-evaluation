#!python
# -*- coding: utf-8 -*-
"""Preprocess audio files."""

import numpy as np
from librosa import load, stft
from librosa.filters import mel
from librosa.effects import trim
import soundfile as sf
from scipy.signal import lfilter
import sox


class AudioProcessor:
    """Process audio data."""

    # signal processing
    trim_method = "vad"
    sample_rate = 16000
    preemph = 0.97
    hop_len = 326
    win_len = 1304
    n_fft = 1304
    n_mels = 80
    f_min = 80

    if trim_method == "vad":
        tfm = sox.Transformer()
        tfm.vad(location=1)
        tfm.vad(location=-1)
        sox_transform = tfm

    @classmethod
    def trim_wav(cls, wav):
        """trim wav"""
        if cls.trim_method == "librosa":
            _, (start_frame, end_frame) = trim(
                wav, top_db=25, frame_length=512, hop_length=128
            )
            start_frame = max(0, start_frame - 0.1 * cls.sample_rate)
            end_frame = min(len(wav), end_frame + 0.1 * cls.sample_rate)

            start = int(start_frame)
            end = int(end_frame)
            if end - start > 1000:  # prevent empty slice
                wav = wav[start:end]
        else:
            wav = cls.sox_transform.build_array(
                input_array=wav, sample_rate_in=cls.sample_rate
            )

        return wav

    @classmethod
    def load_wav(cls, audio_path, is_trim=True) -> np.ndarray:
        """Load and preprocess waveform."""
        wav = load(audio_path, sr=cls.sample_rate)[0]
        wav = wav / (np.abs(wav).max() + 1e-6)
        if is_trim:
            wav = cls.trim_wav(wav)
        return wav

    @classmethod
    def wav2melspectrogram(
        cls,
        x: np.ndarray,
    ) -> np.ndarray:
        """Create a log Mel spectrogram from a raw audio signal."""
        x = lfilter([1, -cls.preemph], [1], x)
        magnitude = np.abs(
            stft(x, n_fft=cls.n_fft, hop_length=cls.hop_len, win_length=cls.win_len)
        )
        mel_fb = mel(cls.sample_rate, cls.n_fft, n_mels=cls.n_mels, fmin=cls.f_min)
        mel_spec = np.dot(mel_fb, magnitude)
        log_mel_spec = np.log(mel_spec + 1e-9)
        return log_mel_spec.T

    @classmethod
    def write_wav(cls, wav, file_path):
        """Write waveform."""
        sf.write(file_path, wav, cls.sample_rate)

    @classmethod
    def file2spectrogram(cls, file_path, return_wav=False):
        """Load audio file and create spectrogram."""

        wav = cls.load_wav(file_path)
        d_mel = cls.wav2melspectrogram(wav)

        if return_wav:
            return wav, d_mel

        return d_mel
