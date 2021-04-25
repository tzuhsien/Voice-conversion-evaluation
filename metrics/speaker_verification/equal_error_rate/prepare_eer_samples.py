"""
    Generate samples for calculating EER.
    The format of data_dirs
        data_dir
            |--- {speaker name}.pkl
    The format of {speaker name}.pkl
        {speaker name}.pkl
            |--- "filename": file name
            |--- "embedding": embedding
"""
import os
from os.path import join as join_path
import random
from argparse import ArgumentParser
import joblib
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("-n", "--n_sample", type=int, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)

    return vars(parser.parse_args())


def generate_sample(metadata, speaker1, speakers, label, nums):
    """
    Calculate cosine similarity.
    Generate positive or negative samples with the label.
    """
    speaker1_embs = random.choices(metadata[speaker1], k=nums)
    speakers_embs = []
    for _ in range(nums):
        speaker = random.choice(speakers)
        speakers_embs.append(random.choice(metadata[speaker]))

    sampels = []
    for speaker1_emb, speakers_emb in zip(speaker1_embs, speakers_embs):
        cosine_similarity = (
            np.inner(speaker1_emb["embedding"], speakers_emb["embedding"])
            / np.linalg.norm(speaker1_emb["embedding"])
            / np.linalg.norm(speakers_emb["embedding"])
        )
        sampels.append((cosine_similarity, label))

    return sampels


def prepare_eer_samples(data_dirs, output_path, n_sample):
    """generate eer samples"""
    metadata = {}
    for data_dir in data_dirs:
        speaker_list = os.listdir(data_dir)
        for speaker in speaker_list:
            metadata[speaker] = joblib.load(join_path(data_dir, speaker))

    samples = []
    speakers = list(metadata.keys())
    for speaker in tqdm(speakers):
        negative_speakers = speakers.copy()
        negative_speakers.remove(speaker)
        samples += generate_sample(metadata, speaker, [speaker], 1, n_sample)
        samples += generate_sample(metadata, speaker, negative_speakers, 0, n_sample)

    joblib.dump(samples, output_path)


if __name__ == "__main__":
    prepare_eer_samples(**parse_args())
