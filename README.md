# MIT-BIH Arrhythmia Classification using Machine Learning

## Overview
This project implements an end-to-end ECG arrhythmia classification pipeline using the MIT-BIH Arrhythmia Database. The workflow includes ECG preprocessing, heartbeat segmentation, feature extraction, and machine learning-based classification.

The goal is to build a reproducible baseline system for heartbeat-level arrhythmia detection using real clinical ECG data.

## Dataset
- **Dataset:** MIT-BIH Arrhythmia Database
- **Source:** PhysioNet
- **Sampling frequency:** 360 Hz

Records were loaded from PhysioNet annotation and waveform files (`.dat`, `.hea`, `.atr`).

## Methodology
1. **ECG preprocessing**
   - Bandpass filtering
   - Z-score normalization

2. **Beat segmentation**
   - Used PhysioNet annotation locations
   - Extracted fixed windows around annotated beats

3. **Label grouping**
   - Grouped beat symbols into 5 classes using an AAMI-inspired mapping:
     - N: Normal
     - S: Supraventricular
     - V: Ventricular
     - F: Fusion
     - Q: Unknown / other

4. **Feature extraction**
   - Mean
   - Standard deviation
   - Minimum / maximum
   - Peak-to-peak amplitude
   - Median
   - Quartiles
   - Skewness
   - Kurtosis
   - Signal energy / variation features

5. **Classification**
   - Random Forest classifier
   - Stratified train-test split

6. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix

## Results
Baseline model performance:

- **Accuracy:** 97.14%
- **Weighted F1-score:** 0.969
- **Macro F1-score:** 0.849

### Observations
The model performed strongly on dominant and clinically relevant classes, especially normal and ventricular beats. Lower recall in minority classes suggests class imbalance and motivates future improvements such as patient-wise evaluation and imbalance-aware training.

## Project Structure
```text
mitbih-arrhythmia-classification/
├── data/
├── results/
├── src/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore