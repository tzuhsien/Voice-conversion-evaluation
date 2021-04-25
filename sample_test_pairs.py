"""Sample test pairs"""
from pathlib import Path
from importlib import import_module
import argparse
import json
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="source dir.")
    parser.add_argument(
        "-scn", "--source_corpus_name", type=str, help="source corpus name."
    )
    parser.add_argument("-t", "--target", type=str, help="target dir.")
    parser.add_argument(
        "-tcn", "--target_corpus_name", type=str, help="target corpus name."
    )
    parser.add_argument(
        "-tn", "--target_number", type=int, default=5, help="number of target samples."
    )
    parser.add_argument(
        "-n", "--sample_number", type=int, default=1000, help="number of samples."
    )
    parser.add_argument("-o", "--output_dir", type=str, help="path of pickle file.")
    parser.add_argument("-p", "--parser_dir", type=str, help="parser name")

    return vars(parser.parse_args())


def create_parser(parser_dir, corpus_name, data_dir):
    """Create parser"""
    parser_path = str(parser_dir / f"{corpus_name}_parser").replace("/", ".")
    parser = getattr(import_module(parser_path), "Parser")

    return parser(data_dir)


def sample_pairs(
    source,
    source_corpus_name,
    target,
    target_corpus_name,
    target_number,
    sample_number,
    output_dir,
    parser_dir,
):
    """Sample pairs"""

    parser_dir = Path(parser_dir)
    source_parser = create_parser(parser_dir, source_corpus_name, source)
    target_parser = create_parser(parser_dir, target_corpus_name, target)

    metadata = {
        "source_random_seed": source_parser.seed,
        "target_random_seed": target_parser.seed,
        "source_corpus": source_corpus_name,
        "target_corpus": target_corpus_name,
        "sample_number": sample_number,
        "target_number": target_number,
        "pairs": [],
    }

    for _ in tqdm(range(sample_number)):
        source_wav, source_speaker_id, content = source_parser.sample_source()
        target_wavs, target_speaker_id = target_parser.sample_targets(target_number)
        metadata["pairs"].append(
            {
                "source_speaker": source_speaker_id,
                "target_speaker": target_speaker_id,
                "src_utt": source_wav,
                "tgt_utts": target_wavs,
                "content": content,
            }
        )

    output_path = (
        Path(output_dir) / f"{source_corpus_name}_to_{target_corpus_name}.json"
    )
    json.dump(metadata, output_path.open("w"), indent=2)


if __name__ == "__main__":
    sample_pairs(**parse_args())
