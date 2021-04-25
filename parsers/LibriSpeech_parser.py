"""LibriSpeech Corpus parser."""
import random
from pathlib import Path, PurePosixPath
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

    def sample_source(self):
        """Sample as source"""
        wav_file = random.choice(self.wav_files)
        speaker_id = self.get_speaker(wav_file)
        content = self.get_content(wav_file)

        return wav_file, speaker_id, content

    def sample_targets(self, number):
        """Sample as target"""
        wav_files = random.choices(self.wav_files, k=number)
        speaker_id = self.get_speaker(wav_files[0])

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
                raise FileNotFoundError("Translation not found for " + wav_name)

        return utterance

    @classmethod
    def get_speaker(cls, file_path):
        """Get speaker for LibriSpeech Corpus."""
        speaker_id = Path(file_path).stem.split("-")[0]

        return speaker_id