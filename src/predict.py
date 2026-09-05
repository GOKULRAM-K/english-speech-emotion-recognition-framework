"""
predict.py - Single-file audio emotion prediction CLI and module.

Reproduces the exact preprocessing pipeline:
Audio -> librosa.load(duration=2.5, offset=0.6) -> extract_features (ZCR + RMSE + MFCCs)
      -> Pad/Truncate to 2376 -> StandardScaler (emotion_scaler.pkl) -> 1D-CNN model -> LabelEncoder (emotion_encoder.pkl) -> Emotion.

Includes graceful missing-weights detection to instruct the user if model weights are absent.
"""

import sys
import os
import argparse
import joblib
import numpy as np
import tensorflow as tf

from features import get_predict_feat


def predict_emotion(audio_path: str, model_path: str = "models/cnn_model_full.keras",
                    scaler_path: str = "models/emotion_scaler.pkl",
                    encoder_path: str = "models/emotion_encoder.pkl") -> str:
    """
    Predicts the emotion of a given audio file using the trained 1D-CNN pipeline.

    Args:
        audio_path (str): Path to input WAV audio file.
        model_path (str): Path to trained Keras model (.keras or .h5).
        scaler_path (str): Path to fitted StandardScaler (.pkl).
        encoder_path (str): Path to fitted LabelEncoder (.pkl).

    Returns:
        str: Predicted emotion label.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"StandardScaler artifact not found at: {scaler_path}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"LabelEncoder artifact not found at: {encoder_path}")

    # Check for model weights file
    if not os.path.exists(model_path):
        print("\n[!] NOTICE: Trained model checkpoint was not found at:", model_path)
        print("    Per repository design, heavy model weights (~86 MB) are not stored directly in this git repo.")
        print("    Please train the model via the notebook or place your trained weights file ('cnn_model_full.keras') in 'models/'.\n")
        sys.exit(1)

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    model = tf.keras.models.load_model(model_path)

    # 1. Feature Extraction & Scaling (exact matching pipeline)
    features = get_predict_feat(audio_path, scaler=scaler, desired_length=2376)

    # 2. 1D-CNN Prediction
    predictions = model.predict(features, verbose=0)

    # 3. Label Decoding
    predicted_emotion = encoder.inverse_transform(predictions)[0]
    return predicted_emotion


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Inference Tool")
    parser.add_argument("--audio", type=str, required=True, help="Path to target WAV audio file")
    parser.add_argument("--model", type=str, default="models/cnn_model_full.keras", help="Path to model weights")
    parser.add_argument("--scaler", type=str, default="models/emotion_scaler.pkl", help="Path to StandardScaler")
    parser.add_argument("--encoder", type=str, default="models/emotion_encoder.pkl", help="Path to LabelEncoder")

    args = parser.parse_args()

    try:
        emotion = predict_emotion(args.audio, args.model, args.scaler, args.encoder)
        print(f"\n[+] Audio: {args.audio}")
        print(f"[+] Predicted Emotion: {emotion}\n")
    except Exception as e:
        print(f"\n[-] Prediction Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
