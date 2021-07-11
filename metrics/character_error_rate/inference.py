"""Character error rate"""
from pathlib import Path

import json
import numpy as np
from tqdm import tqdm
import librosa
import torch
import jiwer
import editdistance as ed

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor


def load_model(root, device):
    """Load model"""
    pretrain_models = {
        "EN": "facebook/wav2vec2-large-960h-lv60-self",
        "DE": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        "FR": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "IT": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        "ES": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    }

    print(f"[INFO]: Load the pre-trained ASR by {pretrain_models[root]}.")
    model = Wav2Vec2ForCTC.from_pretrained(pretrain_models[root]).to(device)
    if root.upper() == "EN":
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            pretrain_models[root]
        )
    elif root.upper() in ["DE", "FR", "IT", "ES"]:
        tokenizer = Wav2Vec2Processor.from_pretrained(
            pretrain_models[root]
        )
    else:
        print(f"{root} not available.")
        exit()

    models = {"model": model, "tokenizer": tokenizer}

    return models


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence


def calculate_character_error_rate(groundtruth, transcription):
    """Calculate character error rate"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    return ed.eval(transcription, groundtruth) / len(groundtruth)


def calculate_score(model, device, data_dir, output_dir, metadata_path, **kwargs):
    """Calculate score"""

    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "evaluation_score.txt"

    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"
    metadata = json.load(Path(metadata_path).open())

    cers = []
    for pair in tqdm(metadata["pairs"]):
        wav, _ = librosa.load(data_dir / pair["converted"], sr=16000)
        groundtruth = pair["content"]

        inputs = model["tokenizer"](
            wav, sampling_rate=16_000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        logits = model["model"](
            input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = model["tokenizer"].batch_decode(predicted_ids)[0]
        cer = calculate_character_error_rate(
            groundtruth, transcription)
        cers.append(cer)

    average_score = np.mean(cers)
    print(f"[INFO]: Average character error rate: {average_score}")
    print(
        f"Average character error rate: {average_score}", file=output_path.open("a"))
