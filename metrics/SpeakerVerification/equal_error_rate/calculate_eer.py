"""
    Computer Equal Error Rate.
    argv[1]: Path of score pairs.
"""
import sys
import joblib

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def calculate_equal_error_rate(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    s = interp1d(fpr, tpr)
    a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    equal_error_rate = brentq(a, 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(equal_error_rate)
    return equal_error_rate, threshold


if __name__ == "__main__":
    samples = joblib.load(sys.argv[1])
    print(f"[INFO]: Number of samples: {len(samples)}")
    scores = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    equal_error_rate, threshold = calculate_equal_error_rate(labels, scores)
    print(f"[INFO]: Equal error rate: {equal_error_rate}")
    print(f"[INFO]: Threshold: {threshold}")
