"""MBNet for MOS prediction"""
from pathlib import Path

import numpy as np
from tqdm import tqdm
import librosa
import sox
import torch

from .model import MBNet


def load_model(root, device):
    """Load model"""

    root = Path(root) / "checkpoints"
    model_paths = sorted(list(root.glob("MBNet*")))
    models = []
    for model_path in model_paths:
        model = MBNet(num_judges=5000)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model.to(device))

    return models


def do_MBNet(model, wavs, device):
    """Do MBNet."""
    mean_scores = 0
    with torch.no_grad():
        for wav in wavs:
            wav = wav.to(device)
            mean_score = model.only_mean_inference(spectrum=wav)
            mean_scores += mean_score.cpu().tolist()[0]

    return mean_scores / len(wavs)


def calculate_score(model, device, data_dir, output_dir, **kwargs):
    """Calculate score"""

    if output_dir is None:
        output_dir = Path(data_dir)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "evaluation_score.txt"

    file_paths = librosa.util.find_files(data_dir)
    tfm = sox.Transformer()
    tfm.norm(-3.0)

    wavs = []
    for file_path in tqdm(file_paths):
        wav, _ = librosa.load(file_path, sr=16000)
        wav = tfm.build_array(input_array=wav, sample_rate_in=16000)
        wav = np.abs(librosa.stft(wav, n_fft=512)).T
        wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        wavs.append(wav)

    mean_scores = []
    for m in tqdm(model):
        mean_score = do_MBNet(m, wavs, device)
        mean_scores.append(mean_score)

    average_score = np.mean(mean_scores)

    print(f"[INFO]: All mean opinion score: {mean_scores}")
    print(f"[INFO]: Average mean opinion score: {average_score}")
    print(
        f"Average mean opinion score: {average_score}", file=output_path.open("a"))
