"""Extract embedding by resemblyzer."""
import os
from os.path import basename, splitext, join as join_path
from argparse import ArgumentParser
import joblib
import librosa
from tqdm import tqdm

from resemblyzer import preprocess_wav, VoiceEncoder


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    return vars(parser.parse_args())


def extract(data_dirs, output_dir):
    """Extract embedding by resemblyzer."""
    encoder = VoiceEncoder()
    os.makedirs(output_dir, exist_ok=True)

    for data_dir in tqdm(data_dirs, position=0):
        speaker_list = [
            speaker_name
            for speaker_name in os.listdir(data_dir)
            if os.path.isdir(join_path(data_dir, speaker_name))
        ]
        for speaker_name in tqdm(speaker_list, position=1, leave=False):
            data = []
            file_list = librosa.util.find_files(join_path(data_dir, speaker_name))
            for file_path in tqdm(file_list, position=2, leave=False):
                wav = preprocess_wav(file_path)
                embedding = encoder.embed_utterance(wav)
                wav_name = splitext(basename(file_path))[0]
                data.append({"filename": wav_name, "embedding": embedding})
            if len(data) == 0:
                continue
            joblib.dump(data, join_path(output_dir, f"{speaker_name}.pkl"))


if __name__ == "__main__":
    extract(**parse_args())
