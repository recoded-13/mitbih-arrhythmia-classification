from __future__ import annotations

from pathlib import Path
from typing import Tuple

import wfdb
import numpy as np


def load_record(record_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load MIT-BIH ECG signal and annotations.

    Returns:
        signal: np.ndarray of shape (n_samples,)
        ann_samples: np.ndarray of annotation sample indices
        fs: sampling frequency
    """
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    signal = record.p_signal[:, 0]  # use first ECG channel
    ann_samples = np.array(annotation.sample)
    ann_symbols = np.array(annotation.symbol)
    fs = record.fs

    return signal, ann_samples, ann_symbols, fs