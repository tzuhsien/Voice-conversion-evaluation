"""MBNet for MOS prediction"""
from pathlib import Path

import json
import yaml
import numpy as np
from tqdm import tqdm

from resemblyzer import preprocess_wav, VoiceEncoder


def load_model(root, device):
    """Load model"""

    model = VoiceEncoder()

    return model


def calculate_score(model, data_dir, output_dir, target_dir, threshold_path, **kwargs):
    """Calculate score"""

    data_dir = Path(data_dir)
    target_dir = Path(target_dir)

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "evaluation_score.txt"

    metadata_path = data_dir / "metadata.json"
    metadata = json.load(metadata_path.open())

    thresholds = yaml.load(Path(threshold_path).open())
    threshold = thresholds[metadata["target_corpus"]]

    n_accept = 0
    for pair in tqdm(metadata["pairs"]):
        wav = preprocess_wav(data_dir / pair["converted"])
        source_emb = model.embed_utterance(wav)

        targets = [target_dir / tgt_utt for tgt_utt in pair["tgt_utts"]]
        target_embs = []
        for target in targets:
            wav = preprocess_wav(target)
            target_embs.append(model.embed_utterance(wav))

        target_emb = np.mean(target_embs, 0)
        cosine_similarity = (
            np.inner(source_emb, target_emb)
            / np.linalg.norm(source_emb)
            / np.linalg.norm(target_emb)
        )
        # print(cosine_similarity)
        if cosine_similarity > threshold:
            n_accept += 1

    svar = n_accept / len(metadata["pairs"])
    print(f"[INFO]: Speaker verification accept rate: {svar}")
    print(
        f"Speaker verification accept rate: {svar}",
        file=output_path.open("a"),
    )
