"""
evaluate.py - Benchmark evaluation and testing routines.

Includes evaluation routines for classification metrics, confusion matrices,
Leave-One-Dataset-Out (LODO) cross-corpus generalization testing, and SNR noise robustness analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model_performance(model, x_test, y_test, class_names=None):
    """
    Evaluates the model on test data, prints classification report, and returns confusion matrix.
    """
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    cm = confusion_matrix(y_true, y_pred)

    return report_df, cm


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plots and optionally saves a heatmap visualization of the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def add_noise(signal, snr_db=10):
    """Adds additive white Gaussian noise to an audio signal at a specified SNR (dB)."""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_signal = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise_signal
