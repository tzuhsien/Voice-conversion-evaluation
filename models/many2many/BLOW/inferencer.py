"""Inferencer of AdaIN-VC"""
from typing import List
from pathlib import Path
import json
import numpy as np
import torch
from torch import Tensor

from audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args_path = checkpoint_dir / "args.pt"
        checkpoint_path = checkpoint_dir / "a.pt"
        speaker2id_path = checkpoint_dir / "speaker2id.json"

        args = torch.load(args_path, map_location=device)
        speaker2id = json.load(speaker2id_path.open())

        # self.model = Model(args.nsqueeze, args.nblocks,
        #                    args.nflows, args.ncha, len(speaker2id), args.lchunk).to(device)
        # self.model.load_state_dict(torch.load(
        #     checkpoint_path))
        self.model = torch.load(checkpoint_path)
        self.model.precalc_matrices('on')
        self.model.eval()

        self.speaker2id = speaker2id
        self.device = device
        self.sample_rate = AudioProcessor.sample_rate
        self.lchunk = args.lchunk
        self.stride = args.lchunk // 2
        self.window = torch.hann_window(args.lchunk).view(1, -1)
        self.frame_energy_thres = 0
        self.ymax = 0.98

    def preprocess(self, wav):
        wav_chucks = []
        for left in range(0, len(wav), self.stride):
            if left+self.lchunk >= len(wav):
                wav_chuck = wav[left:].float()
                wav_chuck = torch.cat([
                    wav_chuck, torch.zeros(self.lchunk - len(wav_chuck))
                ])
            else:
                wav_chuck = wav[left:left+self.lchunk].float()
            frame_energy = (wav_chuck.pow(2).sum() / self.lchunk).sqrt().item()
            if frame_energy >= self.frame_energy_thres:
                wav_chucks.append(wav_chuck)
        return torch.stack(wav_chucks).to(self.device)

    def get_speaker_id(self, speaker_name: str, batch):
        if isinstance(speaker_name, str):
            try:
                speaker_id = self.speaker2id[speaker_name]
            except ValueError:
                print(f"{speaker_name} is not available")
                exit()

        return torch.LongTensor([int(speaker_id)] * batch).to(self.device)

    def flatten(self, chucks):
        wav = torch.zeros((len(chucks)-1)*self.stride+self.lchunk)
        for i, chuck in enumerate(chucks):
            wav[i*self.stride:i*self.stride+self.lchunk] += chuck

        return wav

    def inference(self, src_wav: np.ndarray, source_speaker: str, target_speaker: str) -> Tensor:
        """Inference one utterance."""
        with torch.no_grad():
            src_wav = torch.HalfTensor(src_wav)
            src_wav = self.preprocess(src_wav)
            source_speaker_id = self.get_speaker_id(
                source_speaker, src_wav.size(0))
            target_speaker_id = self.get_speaker_id(
                target_speaker, src_wav.size(0))
            latent = self.model.forward(src_wav, source_speaker_id)[0]
            result = self.model.reverse(latent, target_speaker_id)
            result = result.cpu()
            result *= self.window

        return self.flatten(result)

    def inference_from_path(self, src_path: Path, source_speaker: str, target_speaker: str) -> Tensor:
        """Inference from path."""
        try:
            src_wav = AudioProcessor.file2spectrogram(src_path)
        except ValueError:
            src_wav = AudioProcessor.file2spectrogram(src_path, is_trim=False)

        result = self.inference(src_wav, source_speaker, target_speaker)

        return result

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        source_speaker = pair["source_speaker"]
        target_speaker = pair["target_speaker"]
        result = self.inference_from_path(
            source_utt, source_speaker, target_speaker)

        return result

    def spectrogram2waveform(self, wavs: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        results = []
        for wav in wavs:
            wav = wav.cpu().numpy().astype(np.float32)
            wav -= np.mean(wav)
            mx = np.max(np.abs(wav))
            if mx > 0:
                wav *= self.ymax / mx
            wav = np.array(wav * 32767, dtype=np.int16)
            wav = torch.from_numpy(wav)
            results.append(wav)

        return results
