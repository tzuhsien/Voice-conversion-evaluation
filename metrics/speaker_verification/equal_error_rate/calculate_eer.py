"""
    Computer Equal Error Rate.
    argv[1]: Path of score pairs.
"""
from pathlib import Path
from argparse import ArgumentParser
import joblib

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-n", "--corpus_name", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)

    return vars(parser.parse_args())


def calculate_equal_error_rate(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    def a(x): return 1.0 - x - interp1d(fpr, tpr)(x)
    equal_error_rate = brentq(a, 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(equal_error_rate)
    return equal_error_rate, threshold


def main(data_path, corpus_name, output_path):
    """Main function"""
    samples = joblib.load(data_path)
    print(f"[INFO]: Number of samples: {len(samples)} from {data_path}")
    scores = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    equal_error_rate, threshold = calculate_equal_error_rate(labels, scores)
    print(f"[INFO]: Equal error rate: {equal_error_rate}")
    print(f"[INFO]: Threshold: {threshold}")

    output_path = Path(output_path) / f"{corpus_name}_eer.yaml"
    print(f"Threshold: {threshold}", file=output_path.open("a"))
    print(f"Equal_Error_Rate: {equal_error_rate}", file=output_path.open("a"))


if __name__ == "__main__":
    main(**parse_args())
