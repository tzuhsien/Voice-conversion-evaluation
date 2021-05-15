"""M-AILABS Corpus parser."""
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
import librosa
from librosa.util import find_files
import json


class Parser:
    """Parser"""

    def __init__(self, root):
        seed = random.randint(1, 1000)
        random.seed(seed)

        if (Path(root) / "by_book").exists():
            _root = Path(root) / "by_book"
        else:
            _root = Path(root)

        wav_files = []
        metadata = defaultdict(list)
        speaker_dirs = [
            speaker_dir for speaker_dir in (_root / "female").iterdir() if speaker_dir.is_dir()]
        speaker_dirs += [
            speaker_dir for speaker_dir in (_root / "male").iterdir() if speaker_dir.is_dir()]

        for speaker_dir in speaker_dirs:
            for wav_file in find_files(speaker_dir):
                wav_file = str(PurePosixPath(wav_file).relative_to(root))
                wav_files.append(wav_file)
                speaker_id = self.get_speaker(wav_file)
                metadata[speaker_id].append(wav_file)

        self.root = root
        self.seed = seed
        self.wav_files = wav_files
        self.metadata = metadata

    def set_random_seed(self, seed):
        """Set random seed"""
        random.seed(seed)
        self.seed = seed

    def sample_source(self):
        """Sample as source"""
        content = None
        while content is None:
            wav_file = random.choice(self.wav_files)
            content = self.get_content(wav_file)

        speaker_id = self.get_speaker(wav_file)
        wav, sample_rate = librosa.load(Path(self.root) / wav_file)
        second = len(wav) / sample_rate

        return wav_file, speaker_id, content, second

    def sample_targets(self, number, ignore_id):
        """Sample as target"""

        negative_speakers = list(self.metadata.keys())
        try:
            negative_speakers.remove(ignore_id)
        except ValueError:
            pass
        speaker_id = random.choice(negative_speakers)
        wav_files = random.choices(self.metadata[speaker_id], k=number)

        return wav_files, speaker_id

    def get_content(self, file_path):
        """Get text for VCTK Corpus."""
        file_name = Path(file_path).name.replace("\uf010", "\u0010")

        root = Path(self.root)
        info_path = Path(file_path).parent.parent / "metadata_mls.json"
        info = json.load((root / info_path).open())

        content = info.get(file_name, None)
        if content is None:
            print(file_name)
            return None
        return content["original"]

    def get_speaker_number(self):
        """Get the number of speaker."""
        return len(self.metadata)

    @classmethod
    def get_speaker(cls, file_path):
        """Get speaker for M-AILABS Corpus."""
        return Path(file_path).parent.parent.parent.stem
