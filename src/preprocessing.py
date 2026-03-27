from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = np.std(signal)
    if std == 0:
        return signal.copy()
    return (signal - np.mean(signal)) / std


def segment_beats(
    signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: np.ndarray,
    class_map: dict,
    samples_before: int,
    samples_after: int,
) -> Tuple[np.ndarray, np.ndarray]:
    beats: List[np.ndarray] = []
    labels: List[str] = []

    for sample, symbol in zip(ann_samples, ann_symbols):
        if symbol not in class_map:
            continue

        start = sample - samples_before
        end = sample + samples_after

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        if len(beat) != (samples_before + samples_after):
            continue

        beats.append(beat)
        labels.append(class_map[symbol])

    return np.array(beats), np.array(labels)