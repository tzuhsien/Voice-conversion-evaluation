"""LibriTTS parser."""
import random
from pathlib import Path, PurePosixPath
import librosa
from librosa.util import find_files


class Parser:
    """Parser"""

    def __init__(self, root):
        seed = random.randint(1, 1000)
        random.seed(seed)

        wav_files = [
            str(PurePosixPath(wav_file).relative_to(root))
            for wav_file in find_files(root)
        ]

        self.root = root
        self.seed = seed
        self.wav_files = wav_files

    def set_random_seed(self, seed):
        """Set random seed"""
        random.seed(seed)
        self.seed = seed

    def sample_source(self):
        """Sample as source"""
        wav_file = random.choice(self.wav_files)
        speaker_id = self.get_speaker(wav_file)
        content = self.get_content(wav_file)
        wav, sample_rate = librosa.load(Path(self.root) / wav_file)
        second = len(wav) / sample_rate

        return wav_file, speaker_id, content, second

    def sample_targets(self, number):
        """Sample as target"""
        wav_files = random.choices(self.wav_files, k=number)
        speaker_id = self.get_speaker(wav_files[0])

        return wav_files, speaker_id

    def get_content(self, file_path):
        """Get text for LibriTTS Corpus."""
        file_path = file_path.replace("wav", "normalized.txt")
        file_path = Path(self.root) / file_path
        with file_path.open() as file_text:
            utterance = file_text.readline()

        return utterance

    @classmethod
    def get_speaker(cls, file_path):
        """Get speaker for LibriTTS Corpus."""
        speaker_id = Path(file_path).stem.split("_")[0]

        return speaker_id
