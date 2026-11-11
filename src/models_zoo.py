"""
Model Zoo for Parkinson's Freezing-of-Gait Detection on ESP32.
Contains candidate architectures:
  1. Baseline Tiny 1D-CNN (4.5K params)
  2. Improved Residual 1D-CNN (Res-CNN, ~30K params)
  3. CNN + GRU Hybrid (~25K params)
  4. CNN + LSTM Hybrid (~35K params)
  5. Dilated Temporal Convolutional Network (TCN, ~40K params)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(100, 6), dropout_rate=0.3):
    """Original TinyML 1D-CNN (4,465 params)."""
    inputs = layers.Input(shape=input_shape, name='imu_input')
    x = layers.Conv1D(16, kernel_size=5, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.SeparableConv1D(32, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(32, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='Baseline_1DCNN')

def build_improved_res_cnn(input_shape=(100, 6), dropout_rate=0.3):
    """
    Improved Multi-Scale Residual 1D-CNN (~28K params).
    Uses parallel multi-scale kernels (k=3, k=5) and residual skip connections
    to simultaneously capture fast tremor (4-8 Hz) and cyclic stride breakdown (1-2 Hz).
    """
    inputs = layers.Input(shape=input_shape, name='imu_input')
    
    # Stem
    x = layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 1 (Residual with 32 filters)
    res = x
    b1 = layers.Conv1D(32, kernel_size=3, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)
    b1 = layers.Conv1D(32, kernel_size=3, padding='same', use_bias=False)(b1)
    b1 = layers.BatchNormalization()(b1)
    x = layers.Add()([res, b1])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (50, 32)
    
    # Block 2 (Residual with 64 filters)
    res = layers.Conv1D(64, kernel_size=1, padding='same', use_bias=False)(x)
    res = layers.BatchNormalization()(res)
    
    b2 = layers.Conv1D(64, kernel_size=3, padding='same', use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)
    b2 = layers.Conv1D(64, kernel_size=3, padding='same', use_bias=False)(b2)
    b2 = layers.BatchNormalization()(b2)
    x = layers.Add()([res, b2])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (25, 64)
    
    # Feature aggregation
    x = layers.Conv1D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='Improved_ResCNN')

def build_cnn_gru(input_shape=(100, 6), dropout_rate=0.3):
    """
    Hybrid CNN + GRU (~22K params).
    CNN acts as a spatial-spectral local feature extractor;
    GRU models temporal transitions, cadence decay, and pre-FoG onset progression.
    """
    inputs = layers.Input(shape=input_shape, name='imu_input')
    
    # Conv feature extractor
    x = layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (50, 32)
    
    x = layers.Conv1D(48, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (25, 48)
    
    # Recurrent sequence model
    x = layers.GRU(32, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='CNN_GRU')

def build_cnn_lstm(input_shape=(100, 6), dropout_rate=0.3):
    """
    Hybrid CNN + LSTM (~31K params).
    Uses LSTM cells to maintain long-range memory across the 2.0s window.
    """
    inputs = layers.Input(shape=input_shape, name='imu_input')
    
    x = layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (50, 32)
    
    x = layers.Conv1D(48, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (25, 48)
    
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')

def build_tcn(input_shape=(100, 6), dropout_rate=0.3):
    """
    Dilated Temporal Convolutional Network (TCN, ~35K params).
    Uses dilated convolutions with exponential dilation rates d = 1, 2, 4, 8
    to achieve large receptive field without recurrent overhead.
    100% native operator support in standard TFLM.
    """
    inputs = layers.Input(shape=input_shape, name='imu_input')
    
    # Input projection
    x = layers.Conv1D(32, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Dilated residual blocks (d = 1, 2, 4, 8)
    for dilation_rate in [1, 2, 4, 8]:
        res = x
        f = layers.Conv1D(32, kernel_size=3, padding='same', dilation_rate=dilation_rate, use_bias=False)(x)
        f = layers.BatchNormalization()(f)
        f = layers.ReLU()(f)
        f = layers.Dropout(0.1)(f)
        f = layers.Conv1D(32, kernel_size=3, padding='same', dilation_rate=dilation_rate, use_bias=False)(f)
        f = layers.BatchNormalization()(f)
        x = layers.Add()([res, f])
        x = layers.ReLU()(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='Dilated_TCN')

if __name__ == '__main__':
    for builder in [build_baseline_cnn, build_improved_res_cnn, build_cnn_gru, build_cnn_lstm, build_tcn]:
        m = builder()
        print(f"{m.name:20s}: {m.count_params():,d} parameters")
