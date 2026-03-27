from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLASS_MAP,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    RECORDS,
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    TEST_SIZE,
    VALID_CLASSES,
)
from src.data_loader import load_record
from src.evaluate import evaluate_model
from src.feature_extraction import extract_features
from src.model import build_model
from src.preprocessing import bandpass_filter, segment_beats, zscore_normalize


def ensure_directories() -> None:
    for folder in [DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    all_beats = []
    all_labels = []

    for record_id in RECORDS:
        record_path = DATA_RAW_DIR / record_id
        if not (DATA_RAW_DIR / f"{record_id}.dat").exists():
            print(f"Skipping {record_id}: files not found in {DATA_RAW_DIR}")
            continue

        signal, ann_samples, ann_symbols, fs = load_record(record_path)
        filtered = bandpass_filter(signal, fs=fs)
        normalized = zscore_normalize(filtered)

        beats, labels = segment_beats(
            normalized,
            ann_samples,
            ann_symbols,
            CLASS_MAP,
            SAMPLES_BEFORE,
            SAMPLES_AFTER,
        )

        if len(beats) == 0:
            continue

        all_beats.append(beats)
        all_labels.append(labels)

        print(f"Loaded record {record_id}: {len(beats)} beats")

    if not all_beats:
        raise ValueError("No beats extracted. Check dataset download path and files.")

    X_beats = np.vstack(all_beats)
    y_labels = np.concatenate(all_labels)

    return X_beats, y_labels


def train_pipeline() -> None:
    ensure_directories()

    X_beats, y_labels = build_dataset()

    features = extract_features(X_beats)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    model = build_model()
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test, labels=list(range(len(class_names))), figures_dir=FIGURES_DIR, metrics_dir=METRICS_DIR)

    joblib.dump(model, DATA_PROCESSED_DIR / "rf_model.joblib")
    joblib.dump(le, DATA_PROCESSED_DIR / "label_encoder.joblib")

    pd.DataFrame(X_train).to_csv(DATA_PROCESSED_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(DATA_PROCESSED_DIR / "X_test.csv", index=False)
    pd.DataFrame({"y_train": y_train}).to_csv(DATA_PROCESSED_DIR / "y_train.csv", index=False)
    pd.DataFrame({"y_test": y_test}).to_csv(DATA_PROCESSED_DIR / "y_test.csv", index=False)

    print("Training complete.")
    print("Classes:", class_names)
    print("Saved model, label encoder, metrics, and confusion matrix.")


if __name__ == "__main__":
    train_pipeline()