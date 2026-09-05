# Preprocessing Artifacts & Trained Model Weights

This directory contains pre-fitted label encoding and feature normalization artifacts, as well as documentation regarding model weights.

## Artifact Manifest

| File | Type | Description |
| :--- | :--- | :--- |
| `emotion_scaler.pkl` | `sklearn.preprocessing.StandardScaler` | Fitted scaler for 2,376-dimensional acoustic feature normalization. |
| `emotion_encoder.pkl` | `sklearn.preprocessing.OneHotEncoder` / `LabelEncoder` | Fitted encoder mapping 7 emotion categories (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`). |

## Pretrained Model Weights

The pretrained model checkpoint file (`cnn_model_full.keras` ~86.38 MB / `cnn_model.weights.h5` ~86.37 MB) is **not committed directly to this git repository** due to standard repository size limits and file tracking constraints.

### How to Obtain Model Weights

1. **Local Training**:
   Run the end-to-end training pipeline in `notebooks/speech_emotion_recognition.ipynb`. The script will automatically train the 1D-CNN architecture (7,193,223 parameters) and save `cnn_model_full.keras` directly into `models/`.

2. **External Hosting**:
   If hosting trained weights on an external platform (e.g., Hugging Face Hub or GitHub Releases), place the downloaded `cnn_model_full.keras` file into this `models/` folder.

Once `cnn_model_full.keras` is placed in this directory, `src/predict.py` and evaluation routines will execute seamlessly.
