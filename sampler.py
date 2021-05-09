"""Sample test pairs"""
from pathlib import Path
from importlib import import_module
import argparse
import json
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("source_corpus_name", type=str, help="source corpus name.")
    parser.add_argument("source_dir", type=str, help="source dir.")
    parser.add_argument("target_corpus_name", type=str, help="target corpus name.")
    parser.add_argument("target_dir", type=str, help="target dir.")
    parser.add_argument("-o", "--output_dir", type=str, help="path of pickle file.")
    parser.add_argument("-p", "--parser_dir", type=str, default="./parsers", help="parser name")
    parser.add_argument(
        "-n", "--n_samples", type=int, default=2000, help="number of samples."
    )
    parser.add_argument(
        "-nt",
        "--n_target_samples",
        type=int,
        default=5,
        help="number of target samples.",
    )
    parser.add_argument(
        "--s_seed", type=int, default=None, help="The random seed for sampling source utterances."
    )
    parser.add_argument(
        "--t_seed", type=int, default=None, help="The random seed for sampling target utterances."
    )

    return vars(parser.parse_args())


def create_parser(parser_dir, corpus_name, data_dir):
    """Create parser"""
    parser_path = str(parser_dir / f"{corpus_name}_parser").replace("/", ".")
    parser = getattr(import_module(parser_path), "Parser")

    return parser(data_dir)


def sample_pairs(
    source_corpus_name,
    source_dir,
    target_corpus_name,
    target_dir,
    output_dir,
    parser_dir,
    n_samples,
    n_target_samples,
    s_seed,
    t_seed,
):
    """Sample pairs"""

    parser_dir = Path(parser_dir)
    source_parser = create_parser(parser_dir, source_corpus_name, source_dir)
    target_parser = create_parser(parser_dir, target_corpus_name, target_dir)

    if s_seed is not None:
        source_parser.set_random_seed(s_seed)

    if t_seed is not None:
        target_parser.set_random_seed(t_seed)

    metadata = {
        "source_random_seed": source_parser.seed,
        "target_random_seed": target_parser.seed,
        "source_corpus": source_corpus_name,
        "target_corpus": target_corpus_name,
        "n_samples": n_samples,
        "n_target_samples": n_target_samples,
        "pairs": [],
    }

    for _ in tqdm(range(n_samples)):
        source_wav, source_speaker_id, content, second = source_parser.sample_source()
        target_wavs, target_speaker_id = target_parser.sample_targets(n_target_samples)
        metadata["pairs"].append(
            {
                "source_speaker": source_speaker_id,
                "target_speaker": target_speaker_id,
                "src_utt": source_wav,
                "tgt_utts": target_wavs,
                "content": content,
                "src_second": second,
            }
        )

    metadata["pairs"].sort(key=lambda x: x["src_second"], reverse=True)

    output_path = (
        Path(output_dir) / f"{source_corpus_name}_to_{target_corpus_name}.json"
    )
    json.dump(metadata, output_path.open("w"), indent=2)


if __name__ == "__main__":
    sample_pairs(**parse_args())
