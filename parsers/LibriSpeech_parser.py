"""LibriSpeech Corpus parser."""
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
        wav_file = random.choice(self.wav_files)
        speaker_id = self.get_speaker(wav_file)
        content = self.get_content(wav_file)
        wav, _ = librosa.load(Path(self.root) / wav_file)
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
        """Get text for LibriSpeech Corpus."""
        wav_name = Path(file_path).stem
        speaker_id, chapter_id, _ = wav_name.split("-")
        file_name = speaker_id + "-" + chapter_id + ".trans.txt"
        file_path = Path(self.root) / speaker_id / chapter_id / file_name
        with file_path.open() as file_text:
            for line in file_text:
                fileid_text, utterance = line.strip().split(" ", 1)
                if wav_name == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError(
                    "Translation not found for " + wav_name)

        return utterance

    def get_speaker_number(self):
        """Get the number of speaker."""
        return len(self.metadata)

    @classmethod
    def get_speaker(cls, file_path):
        """Get speaker for LibriSpeech Corpus."""
        speaker_id = Path(file_path).stem.split("-")[0]

        return speaker_id
