from os.path import splitext
import joblib
import numpy as np
import json
from argparse import ArgumentParser


def verify_speaker(metadata_path, source_path, target_path, threshold, output_dir):
    metadata = json.load(open(metadata_path, "r"))
    sources = joblib.load(source_path)
    targets = joblib.load(target_path)

    count = 0
    for pair in metadata:
        emb_s = sources[splitext(pair[0])[0]]
        emb_t = targets[pair[1]]
        cosine_similarity = (
            np.inner(emb_s, emb_t) / np.linalg.norm(emb_s) / np.linalg.norm(emb_t)
        )
        # print(cosine_similarity)
        if cosine_similarity > threshold:
            count += 1

    print(f"Accuracy: {count / len(metadata)}")


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("metadata_path", type=str)
    PARSER.add_argument("-s", "--source_path", type=str, required=True)
    PARSER.add_argument("-t", "--target_path", type=str, required=True)
    PARSER.add_argument("--threshold", type=float, required=True)
    PARSER.add_argument("-o", "--output_dir", type=str, required=True)
    verify_speaker(**vars(PARSER.parse_args()))
