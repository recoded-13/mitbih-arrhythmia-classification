from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis


def extract_features(beats: np.ndarray) -> np.ndarray:
    """
    Simple statistical feature extractor for classical ML baseline.

    Input:
        beats: shape (n_beats, n_samples)
    Output:
        features: shape (n_beats, n_features)
    """
    features = []
    for beat in beats:
        feat = [
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            np.ptp(beat),
            np.median(beat),
            np.percentile(beat, 25),
            np.percentile(beat, 75),
            skew(beat),
            kurtosis(beat),
            np.sum(np.abs(np.diff(beat))),
            np.sqrt(np.mean(np.square(beat))),
        ]
        features.append(feat)

    return np.array(features)