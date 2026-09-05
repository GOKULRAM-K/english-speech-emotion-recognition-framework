"""
dataset.py - Dataset ingestion and dataframe construction utilities for RAVDESS, CREMA-D, TESS, and SAVEE corpora.

This module provides functions to parse raw audio dataset directories and unify them into a harmonized pandas DataFrame.
"""

import os
import pandas as pd


def load_ravdess(ravdess_path: str) -> pd.DataFrame:
    """Parses RAVDESS directory and maps filename tags to emotion labels."""
    file_emotion = []
    file_path = []
    if not os.path.exists(ravdess_path):
        return pd.DataFrame(columns=['Path', 'Emotions'])

    for actor_dir in os.listdir(ravdess_path):
        actor_path = os.path.join(ravdess_path, actor_dir)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    part = file.split('.')[0].split('-')
                    if len(part) >= 3:
                        file_emotion.append(int(part[2]))
                        file_path.append(os.path.join(actor_path, file))

    df = pd.DataFrame({'Path': file_path, 'Emotions': file_emotion})
    emotion_map = {
        1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
    }
    df['Emotions'] = df['Emotions'].map(emotion_map)
    return df


def load_cremad(cremad_path: str) -> pd.DataFrame:
    """Parses CREMA-D directory and maps filename codes to emotion labels."""
    file_emotion = []
    file_path = []
    if not os.path.exists(cremad_path):
        return pd.DataFrame(columns=['Path', 'Emotions'])

    for file in os.listdir(cremad_path):
        if file.endswith('.wav'):
            file_path.append(os.path.join(cremad_path, file))
            part = file.split('_')
            if len(part) >= 3:
                code = part[2]
                if code == 'SAD':
                    file_emotion.append('sad')
                elif code == 'ANG':
                    file_emotion.append('angry')
                elif code == 'DIS':
                    file_emotion.append('disgust')
                elif code == 'FEA':
                    file_emotion.append('fear')
                elif code == 'HAP':
                    file_emotion.append('happy')
                elif code == 'NEU':
                    file_emotion.append('neutral')
                else:
                    file_emotion.append('Unknown')
            else:
                file_emotion.append('Unknown')

    return pd.DataFrame({'Path': file_path, 'Emotions': file_emotion})


def load_tess(tess_path: str) -> pd.DataFrame:
    """Parses TESS directory and maps filename strings to emotion labels."""
    file_emotion = []
    file_path = []
    if not os.path.exists(tess_path):
        return pd.DataFrame(columns=['Path', 'Emotions'])

    for dir_name in os.listdir(tess_path):
        dir_full = os.path.join(tess_path, dir_name)
        if os.path.isdir(dir_full):
            for file in os.listdir(dir_full):
                if file.endswith('.wav'):
                    part = file.split('.')[0].split('_')
                    if len(part) >= 3:
                        tag = part[2]
                        if tag == 'ps':
                            file_emotion.append('surprise')
                        else:
                            file_emotion.append(tag)
                        file_path.append(os.path.join(dir_full, file))

    return pd.DataFrame({'Path': file_path, 'Emotions': file_emotion})


def load_savee(savee_path: str) -> pd.DataFrame:
    """Parses SAVEE directory and maps filename codes to emotion labels."""
    file_emotion = []
    file_path = []
    if not os.path.exists(savee_path):
        return pd.DataFrame(columns=['Path', 'Emotions'])

    for file in os.listdir(savee_path):
        if file.endswith('.wav'):
            file_path.append(os.path.join(savee_path, file))
            part = file.split('_')[1]
            ele = part[:-6]
            if ele == 'a':
                file_emotion.append('angry')
            elif ele == 'd':
                file_emotion.append('disgust')
            elif ele == 'f':
                file_emotion.append('fear')
            elif ele == 'h':
                file_emotion.append('happy')
            elif ele == 'n':
                file_emotion.append('neutral')
            elif ele == 'sa':
                file_emotion.append('sad')
            else:
                file_emotion.append('surprise')

    return pd.DataFrame({'Path': file_path, 'Emotions': file_emotion})


def build_consolidated_dataset(ravdess_dir: str, cremad_dir: str, tess_dir: str, savee_dir: str) -> pd.DataFrame:
    """Combines all four speech emotion datasets into a single consolidated dataframe."""
    ravdess_df = load_ravdess(ravdess_dir)
    cremad_df = load_cremad(cremad_dir)
    tess_df = load_tess(tess_dir)
    savee_df = load_savee(savee_dir)

    consolidated = pd.concat([ravdess_df, cremad_df, tess_df, savee_df], axis=0, ignore_index=True)
    return consolidated
