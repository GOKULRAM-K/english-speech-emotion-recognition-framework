"""
model.py - 1D Convolutional Neural Network (1D-CNN) architecture definition.

Implements the 5-block 1D-CNN proposed for Speech Emotion Recognition (SER),
featuring Batch Normalization, Max Pooling, Dropout (0.2), and Dense layers.
Total trainable parameters: 7,193,223.
"""

import tensorflow as tf
from tensorflow.keras import layers as L


def build_1d_cnn_model(input_shape=(2376, 1), num_classes=7) -> tf.keras.Model:
    """
    Constructs and compiles the 5-Block 1D-CNN classifier for Speech Emotion Recognition.

    Args:
        input_shape (tuple): Dimension of the input feature vector, default (2376, 1).
        num_classes (int): Number of emotion categories, default 7.

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    model = tf.keras.Sequential([
        # Block 1
        L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=5, strides=2, padding='same'),

        # Block 2
        L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=5, strides=2, padding='same'),
        L.Dropout(0.2),

        # Block 3
        L.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=5, strides=2, padding='same'),

        # Block 4
        L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=5, strides=2, padding='same'),
        L.Dropout(0.2),

        # Block 5
        L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3, strides=2, padding='same'),
        L.Dropout(0.2),

        # Fully Connected Head
        L.Flatten(),
        L.Dense(512, activation='relu'),
        L.BatchNormalization(),
        L.Dense(num_classes, activation='softmax')
    ], name="SER_1D_CNN")

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
