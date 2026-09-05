"""
features.py - Feature extraction and audio augmentation routines.

Extracts frame-level Zero Crossing Rate (ZCR), Root Mean Square Energy (RMSE),
and Mel-Frequency Cepstral Coefficients (MFCCs), forming a concatenated
2,376-dimensional acoustic feature representation per audio sample.
"""

import librosa
import numpy as np


def zcr(data, frame_length=2048, hop_length=512):
    """Calculates Zero Crossing Rate across frames."""
    zcr_feature = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_feature)


def rmse(data, frame_length=2048, hop_length=512):
    """Calculates Root Mean Square Energy across frames."""
    rmse_feature = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_feature)


def mfcc(data, sr=22050, frame_length=2048, hop_length=512, flatten=True):
    """Calculates MFCC coefficients across frames."""
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """
    Concatenates Zero Crossing Rate, Root Mean Square Energy, and MFCCs into a single 1D feature array.
    """
    result = np.array([])
    result = np.hstack((
        result,
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return result


def noise(data, noise_factor=0.035):
    """Adds random Gaussian noise for data augmentation."""
    noise_amp = noise_factor * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def pitch(data, sr, pitch_factor=0.7):
    """Applies pitch shifting for data augmentation."""
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)


def get_features(path, duration=2.5, offset=0.6):
    """
    Extracts features for original, noised, pitched, and pitched+noised audio signals.
    Returns stacked feature arrays used during dataset preprocessing.
    """
    data, sr = librosa.load(path, duration=duration, offset=offset)

    aud = extract_features(data, sr=sr)
    audio = np.array(aud)

    noised_audio = noise(data)
    aud2 = extract_features(noised_audio, sr=sr)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio, sr=sr)
    audio = np.vstack((audio, aud3))

    pitched_audio1 = pitch(data, sr)
    pitched_noised_audio = noise(pitched_audio1)
    aud4 = extract_features(pitched_noised_audio, sr=sr)
    audio = np.vstack((audio, aud4))

    return audio


def get_predict_feat(path, scaler=None, desired_length=2376):
    """
    Extracts, standardizes (truncate/pad to desired_length=2376), scales, and shapes
    an individual audio file for 1D-CNN inference.
    """
    d, sr = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d, sr=sr)

    current_length = len(res)
    if current_length > desired_length:
        res = res[:desired_length]
    elif current_length < desired_length:
        res = np.pad(res, (0, desired_length - current_length), 'constant')

    result = np.reshape(res, (1, desired_length))
    if scaler is not None:
        result = scaler.transform(result)

    final_result = np.expand_dims(result, axis=2)
    return final_result
