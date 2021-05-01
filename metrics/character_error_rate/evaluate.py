#!/usr/bin/env python3
"""
    Evaluate character error rate and word error rate.
"""
import os
import argparse
from multiprocessing import Pool
import numpy as np
import librosa
import editdistance as ed

from google_asr import Google_ASR
from normalizer import Normalizer


def compute_cer(transcript, groundtruth):
    """Computer character error rate."""

    return ed.eval(transcript, groundtruth) / len(groundtruth)


def compute_wer(transcript, groundtruth):
    """Computer word error rate."""

    # split by space and filter None
    transcript = list(filter(None, transcript.split()))
    groundtruth = list(filter(None, groundtruth.split()))

    return ed.eval(transcript, groundtruth) / len(groundtruth)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-r", type=str,
                        help="path to directory of data.")
    parser.add_argument("--sample_rate", "-s", type=int,
                        help="sample rate of wav")
    parser.add_argument("--language", "-l", type=str, default='en-US',
                        help="language of wav (en-US or cmn-Hans-CN)")
    parser.add_argument("--n_thread", "-n", type=int, default=4,
                        help="number of thread")

    return parser.parse_args()


def evaluate(asr, normalizer, wav_path, language):
    """evaluate wavfile by character error rate and word error rate"""

    txt_path = f"{os.path.splitext(wav_path)[0]}.txt"
    groundtruth = open(txt_path, 'r').readline().strip('\r\n')
    transcript_path = f"{os.path.splitext(wav_path)[0]}_transcript.txt"
    if os.path.exists(transcript_path):
        transcript = open(transcript_path, 'r').readline().strip('\r\n')
    else:
        transcript = asr.recognize(wav_path)

    if language == 'cmn-Hans-CN':
        transcript = normalizer.normalize_sentence_cn(transcript)
        groundtruth = normalizer.normalize_sentence_cn(groundtruth)
        cer = compute_wer(transcript, groundtruth)
        wer = cer
        return cer, wer

    transcript = normalizer.normalize_sentence(transcript)
    groundtruth = normalizer.normalize_sentence(groundtruth)
    cer = compute_cer(transcript, groundtruth)
    wer = compute_wer(transcript, groundtruth)
    return cer, wer


if __name__ == "__main__":
    args = parse_args()
    asr = Google_ASR(args.language, args.sample_rate)
    normalizer = Normalizer()

    wav_list = librosa.util.find_files(args.root_dir)
    data = [(asr, normalizer, wav_path, args.language) for wav_path in wav_list]
    pool = Pool(args.n_thread)
    results = pool.starmap(evaluate, data)
    average_cer = np.mean([result[0] for result in results]) * 100
    average_wer = np.mean([result[1] for result in results]) * 100
    print(f"average character error rate: {average_cer:.2f}")
    print(f"average word error rate: {average_wer:.2f}")
