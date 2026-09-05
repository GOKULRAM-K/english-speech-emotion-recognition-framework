# Cross-Corpus Robust Speech Emotion Recognition (SER)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end, deep learning framework for **Speech Emotion Recognition (SER)** across multiple acoustic domains. Featuring a **5-Block 1D Convolutional Neural Network (1D-CNN)** operating on frame-level acoustic feature representations ($ZCR \parallel RMSE \parallel \text{MFCCs}$), this repository provides comprehensive evaluation routines for cross-corpus generalization (Leave-One-Dataset-Out / LODO), fine-tuning adaptation, and SNR noise robustness.

---

## Key Highlights

- **Consolidated Test Metric**: Achieves **98.03% classification accuracy** and **0.980 Macro-F1** on the consolidated held-out test set (**9,730 test samples**) across four harmonized speech-emotion corpora.
- **Harmonized Corpora**: Integrates audio samples from **CREMA-D**, **RAVDESS**, **SAVEE**, and **TESS**, spanning 7 core emotion categories: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
- **1D-CNN Architecture**: Compact 5-block 1D-CNN model architecture with **7,193,223 trainable parameters** (82.38 MB disk size).
- **Cross-Corpus Transfer & LODO**: Rigorous Leave-One-Dataset-Out testing demonstrating cross-domain adaptation and fine-tuning gains.
- **Noise Robustness**: Evaluated against additive noise across varying Signal-to-Noise Ratios (10 dB SNR).

---

## Model Architecture & Feature Representation

### 1D-CNN Classifier Architecture

The classifier is a 5-block **1D Convolutional Neural Network (1D-CNN)** designed to capture temporal acoustic patterns across concatenated frame features.

![1D-CNN Architecture](assets/cnn_architecture.png)

```
Input Feature Vector (1, 2376, 1)
  │
  ├── [Conv1D (512, k=5, s=1)] ➔ BatchNorm ➔ MaxPool1D (p=5, s=2)
  ├── [Conv1D (512, k=5, s=1)] ➔ BatchNorm ➔ MaxPool1D (p=5, s=2) ➔ Dropout (0.2)
  ├── [Conv1D (256, k=5, s=1)] ➔ BatchNorm ➔ MaxPool1D (p=5, s=2)
  ├── [Conv1D (256, k=3, s=1)] ➔ BatchNorm ➔ MaxPool1D (p=5, s=2) ➔ Dropout (0.2)
  ├── [Conv1D (128, k=3, s=1)] ➔ BatchNorm ➔ MaxPool1D (p=3, s=2) ➔ Dropout (0.2)
  │
  ├── Flatten
  ├── Dense (512, ReLU) ➔ BatchNorm
  └── Dense (7, Softmax) ➔ Output Emotion Probabilities
```

### Acoustic Feature Extraction Pipeline

The model operates on a **2,376-dimensional concatenated acoustic feature vector** extracted per audio sample:

$$\text{Feature Vector} = \big[ \text{ZCR} \parallel \text{RMSE} \parallel \text{MFCCs} \big] \in \mathbb{R}^{2376}$$

- **Zero Crossing Rate (ZCR)**: Captures high-frequency acoustic content and unvoiced speech boundaries.
- **Root Mean Square Energy (RMSE)**: Captures frame-level signal energy and intensity variations.
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: Captures spectral power distribution and timbral characteristics.

> **Acoustic Interpretation Note**: Mel-spectrogram visualizations (below) are provided for acoustic interpretation and visual demonstration. The classifier itself processes the 2,376-dimensional frame feature representation.

![Mel Spectrogram Visualization](assets/figure1_mel_spectrogram.png)

> **Implementation Note on Audio Duration**: The experimental code executes `librosa.load(path, duration=2.5, offset=0.6)` with feature padding/truncation to fixed length $N = 2376$, whereas the paper text describes a 2-second standardized representation ($32 \text{ features/frame} \times 75 \text{ frames} = 2,376$). The working pipeline code in `src/features.py` preserves the exact audited notebook feature pipeline responsible for the reported results.

---

## Experimental Results

### 1. Training & Validation Performance

The model converged smoothly over 50 epochs, achieving stability without severe overfitting due to Batch Normalization and Dropout regularization.

![Training & Validation Curves](assets/figure3_training_curves.png)

### 2. Consolidated Test Set Performance (9,730 Samples)

Evaluated on the consolidated held-out test split (9,730 samples), the model achieved an overall accuracy of **98.03%** and a **0.980 Macro-F1 score**.

| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Angry** | 0.975 | 0.982 | 0.979 | 1,484 |
| **Disgust** | 0.982 | 0.972 | 0.977 | 1,558 |
| **Fear** | 0.973 | 0.981 | 0.977 | 1,505 |
| **Happy** | 0.977 | 0.970 | 0.973 | 1,619 |
| **Neutral** | 0.983 | 0.988 | 0.985 | 1,558 |
| **Sad** | 0.984 | 0.985 | 0.984 | 1,478 |
| **Surprise** | 0.992 | 0.981 | 0.987 | 528 |
| **Overall Accuracy** | **0.980** | **0.980** | **0.980** | **9,730** |

![Consolidated Confusion Matrix](assets/figure4_confusion_matrix_pretty.png)

### 3. Feature Embedding Representation (t-SNE) & ROC Analysis

t-SNE projection of the penultimate dense layer (512-d) demonstrates distinct, well-separated clusters for each emotion class. Multi-class ROC analysis shows Area Under Curve (AUC) $\approx 0.99+$ across all 7 emotions.

| Penultimate Layer t-SNE | Multi-Class ROC Curves |
| :---: | :---: |
| ![t-SNE Embeddings](assets/figure5_tsne.png) | ![ROC Curves](assets/figure7_roc.png) |

---

## Cross-Corpus Transfer (LODO) & Noise Robustness

### Leave-One-Dataset-Out (LODO) Generalization

To test cross-domain generalization, the model was trained on three corpora and evaluated on a completely unseen fourth corpus (LODO). Fine-tuning on a small target-domain split significantly improves transfer performance.

| Held-Out Dataset | Scratch Accuracy (%) | Fine-Tuned Accuracy (%) | $\Delta$ Accuracy | Paired $t$-test $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| **SAVEE** | 18.75% | 23.80% | **+5.05%** | $2.54 \times 10^{-5}$ |
| **RAVDESS** | 19.91% | 21.23% | **+1.32%** | $3.12 \times 10^{-4}$ |
| **TESS** | 19.70% | 23.87% | **+4.17%** | $3.32 \times 10^{-5}$ |
| **CREMA-D** | 22.09% | 21.95% | -0.14% | $6.13 \times 10^{-4}$ |

### Noise Robustness Benchmark

Evaluating model resilience against additive white Gaussian noise at 10 dB SNR highlights the importance of noise-augmented training.

![Noise Robustness Curve](assets/figure_noise_curve.png)

---

## Directory Layout

```
.
├── README.md                           # Publication README with metrics & figures
├── requirements.txt                    # Python environment dependencies
├── .gitignore                          # Excludes PDFs, heavy weights (*.keras), & audio
├── LICENSE                             # MIT Open Source License
│
├── notebooks/
│   └── speech_emotion_recognition.ipynb# Cleaned end-to-end research notebook
│
├── src/                                # Audited modular Python source code
│   ├── dataset.py                      # Dataset loading for CREMA-D, RAVDESS, SAVEE, TESS
│   ├── features.py                     # ZCR + RMSE + MFCC feature extraction (2,376-d)
│   ├── model.py                        # 5-Block 1D-CNN architecture definition
│   ├── evaluate.py                     # Classification metrics & LODO benchmarking
│   └── predict.py                      # Single-file audio emotion inference CLI
│
├── assets/                             # Publication figures
├── metrics/                            # Quantitative metric CSV tables
└── models/                             # Encoders, scalers, and weights guide
    ├── emotion_encoder.pkl             # Fitted LabelEncoder
    ├── emotion_scaler.pkl              # Fitted StandardScaler
    └── README.md                       # Model weights loading & saving guide
```

---

## Quick Start & Installation

### 1. Environment Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/GOKULRAM-K/english-speech-emotion-recognition.git
cd english-speech-emotion-recognition
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the four publicly available datasets and arrange them as follows:
- **CREMA-D**: [Kaggle CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)
- **RAVDESS**: [Kaggle RAVDESS](https://www.kaggle.com/datasets/uwrf/ravdess-emotional-speech-audio)
- **SAVEE**: [Kaggle SAVEE](https://www.kaggle.com/datasets/ejlok1/savee-database)
- **TESS**: [Kaggle TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

### 3. Notebook Execution

Open and run `notebooks/speech_emotion_recognition.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab for step-by-step training and visualization.

### 4. Single-Audio Inference

To run emotion prediction on a custom audio file using `src/predict.py`:

```bash
python src/predict.py --audio path/to/sample.wav --model models/cnn_model_full.keras
```

> **Note on Model Weights**: Heavy model weights (`cnn_model_full.keras` ~86 MB) are not stored directly in this repository due to file-size limits. Please train the model via the notebook or place your trained weights file into `models/`. See [`models/README.md`](models/README.md) for detailed instructions.

---

## Author & Contact

**Gokul Ram K**
- **Email**: [gokul.ram.kannan210905@gmail.com](mailto:gokul.ram.kannan210905@gmail.com)
- **LinkedIn**: [Gokul Ram K](https://www.linkedin.com/in/gokul-ram-k-277a6a308/)

---

## License

Distributed under the [MIT License](LICENSE).
