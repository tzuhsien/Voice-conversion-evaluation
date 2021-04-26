"""MBNet for MOS prediction"""
from pathlib import Path

import json
import yaml
import numpy as np
from tqdm import tqdm

from resemblyzer import preprocess_wav, VoiceEncoder


def load_model(root):
    """Load model"""

    model = VoiceEncoder()

    return model


def calculate_score(
    model, device, data_dir, target_dir, threshold_path, output, **kwargs
):
    """Calculate score"""

    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.json"
    metadata = json.load(metadata_path.open())

    thresholds = yaml.load(open(threshold_path, "r"))
    threshold = thresholds[metadata["target_corpus"]]
    count = 0

    for pair in tqdm(metadata["pairs"]):
        wav = preprocess_wav(data_dir / pair["converted"])
        source_emb = model.embed_utterance(wav)

        targets = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
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
            count += 1
    svar = count / len(metadata["pairs"])
    print(f"[INFO]: Speaker verification accept rate: {svar}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Speaker verification accept rate: {svar}",
        file=output.open("a"),
    )
