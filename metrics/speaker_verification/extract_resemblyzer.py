"""Extract embedding by resemblyzer."""
import os
from os.path import basename, splitext, join as join_path
from argparse import ArgumentParser
import joblib
import librosa
from tqdm import tqdm

from resemblyzer import preprocess_wav, VoiceEncoder


def extract(data_dirs, output_dir):
    """Extract embedding by resemblyzer."""
    encoder = VoiceEncoder()

    data = {}
    for data_dir in tqdm(data_dirs, position=0):
        file_list = librosa.util.find_files(data_dir)
        for file_path in tqdm(file_list, position=1, leave=False):
            wav = preprocess_wav(file_path)
            embedding = encoder.embed_utterance(wav)
            wav_name = splitext(basename(file_path))[0]
            data[wav_name] = embedding

    joblib.dump(data, f"{output_dir}.pkl")


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("data_dirs", type=str, nargs="+")
    PARSER.add_argument("-o", "--output_dir", type=str, required=True)
    extract(**vars(PARSER.parse_args()))
