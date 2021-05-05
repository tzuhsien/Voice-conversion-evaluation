"""MBNet for MOS prediction"""
import re
from pathlib import Path

import json
import numpy as np
from tqdm import tqdm
import librosa
import torch
import inflect
import jiwer
import editdistance as ed

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


def load_model(root, device):
    """Load model"""

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(
        device
    )
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self"
    )

    models = {"model": model, "tokenizer": tokenizer}

    return models


def normalize_sentence(sentence, digit2en):
    """Normalize sentence"""
    # Convert all characters to lower.
    sentence = sentence.lower()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Convert digit into english.
    digits = re.findall(r"\d+", sentence)
    for digit in digits:
        sentence = sentence.replace(digit, digit2en.number_to_words(digit), 1)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to lower.
    sentence = sentence.lower()

    return sentence


def calculate_character_error_rate(groundtruth, transcription, digit2en):
    """Calculate character error rate"""
    groundtruth = normalize_sentence(groundtruth, digit2en)
    transcription = normalize_sentence(transcription, digit2en)

    return ed.eval(transcription, groundtruth) / len(groundtruth)


def calculate_score(model, device, data_dir, output_dir, **kwargs):
    """Calculate score"""

    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "evaluation_score.txt"

    metadata_path = data_dir / "metadata.json"
    metadata = json.load(metadata_path.open())

    digit2en = inflect.engine()

    cers = []
    for pair in tqdm(metadata["pairs"]):
        wav, _ = librosa.load(data_dir / pair["converted"], sr=16000)
        groundtruth = pair["content"]

        inputs = model["tokenizer"](wav, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        logits = model["model"](input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = model["tokenizer"].batch_decode(predicted_ids)[0]
        cer = calculate_character_error_rate(groundtruth, transcription, digit2en)
        cers.append(cer)

    average_score = np.mean(cers)
    print(f"[INFO]: Average character error rate: {average_score}")
    print(f"Average character error rate: {average_score}", file=output_path.open("a"))