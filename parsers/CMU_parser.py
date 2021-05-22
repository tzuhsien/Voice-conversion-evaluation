"""CMU_ARCTIC Corpus parser."""
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
import librosa
from librosa.util import find_files


class Parser:
    """Parser"""

    def __init__(self, root):
        seed = random.randint(1, 1000)
        random.seed(seed)

        wav_files = []
        metadata = defaultdict(list)
        speaker_dirs = [
            speaker_dir for speaker_dir in Path(root).iterdir() if speaker_dir.is_dir()]

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
            speaker_id = self.get_speaker(wav_file)
            content = self.get_content(wav_file)
        wav, sample_rate = librosa.load(Path(self.root) / wav_file)
        wav, _ = librosa.effects.trim(wav)
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

    def load_content(self, file_path):
        """Read content"""
        content_info = {}

    def get_content(self, file_path):
        """Get text for CMU_ARCTIC Corpus."""
        wav_name = Path(file_path).stem
        speaker_id = self.get_speaker(file_path)

        context_path = Path(self.root) / speaker_id / "etc/txt.done.data"
        with context_path.open() as text_file:
            for line in text_file:
                utterance_id, utterance = line.strip().split(" ", 2)[1:]
                if utterance_id == wav_name:
                    return utterance[1:-3].lower()

        return None

    def get_speaker_number(self):
        """Get the number of speaker."""
        return len(self.metadata)

    @classmethod
    def get_speaker(cls, file_path):
        """Get speaker for CMU_ARCTIC Corpus."""
        speaker_id = file_path.split("/")[0]

        return speaker_id
