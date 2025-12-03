"""ML helpers: synthetic dataset generation and simple Keras models.

This module provides small utilities to generate simulated datasets and a
compact dense Keras model for parameter estimation.
"""

from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ..classes.archs.superkn import get_output_fields_jit

try:  # TensorFlow is optional
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # type: ignore

def parameter_grid(N: int, D: int, a: float, b: float) -> np.ndarray:
    """Generate a regular grid [a, b]^D with N steps per axis.

    Args:
        N: Resolution per axis (points per dimension).
        D: Parameter space dimensionality.
        a: Lower bound.
        b: Upper bound.

    Returns:
        np.ndarray of shape (N**D, D) listing all grid points.
    """
    return np.array([a + (b - a) * (x // N ** np.arange(D, dtype=float) % N / N) for x in range(N ** D)])

def parameter_basis(D: int, b: float = 1.0) -> np.ndarray:
    """Canonical basis augmented with the zero vector in R^D.

    Args:
        D: Dimension.
        b: Norm of each basis vector (default 1).

    Returns:
        np.ndarray of shape (D+1, D).
    """
    vectors = np.zeros((D + 1, D))
    for i in range(D):
        vectors[i + 1, i] = b
    return vectors

def parameter_basis_2p(D: int, b: float = 1.0) -> np.ndarray:
    """Two-point basis: {0, b·e_i, 2b·e_i} for i=1..D.

    Args:
        D: Dimension.
        b: Basis step.

    Returns:
        np.ndarray of shape (2D+1, D).
    """
    vectors = np.zeros((2 * D + 1, D))
    for i in range(D):
        vectors[2 * i + 1, i] = b
        vectors[2 * i + 2, i] = 2 * b
    return vectors

def get_dataset(size: int = 1000, λ: float = 1.65e-6, λ0: float = 1.65e-6) -> np.ndarray:
    """Build a structured synthetic dataset.

    Args:
        size: Number of samples to generate.
        λ: Wavelength in meters (default 1.65e-6).
        λ0: Reference wavelength in meters (default 1.65e-6).

    Returns:
        np.ndarray of shape (size, vector_len).
    """
    # Grid points for applied shifts (φ)
    # Using step λ/3 as in original code (1.65/3 -> λ/3)
    grid_points = parameter_basis_2p(14, λ / 3)
    
    # Vector length: (outputs per point) * (number of points) + (targets)
    # Outputs: 6 darks + 1 bright = 7
    # Targets: 14 intrinsic errors (σ)
    vector_len = len(grid_points) * 7 + 14
    dataset = np.empty((size, vector_len))
    
    # Input fields (Star signals) - normalized
    ψ = np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j], dtype=complex)
    output_order = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)

    for v in range(size):
        # shifts_total_opd corresponds to σ (intrinsic errors)
        # Random errors up to λ/10
        σ = np.random.uniform(0, 1, 14) * λ / 10
        
        vector = np.empty(vector_len)
        for (p, point) in enumerate(grid_points):
            # point corresponds to φ (applied shifts)
            φ = point
            
            # Get complex output fields
            # get_output_fields_jit returns ψout[output_order]
            # ψout has 7 elements: Bright, D1a, D1b, D2a, D2b, D3a, D3b
            # If output_order is [0..6], then:
            # 0: Bright
            # 1,2: Dark 1 pair
            # 3,4: Dark 2 pair
            # 5,6: Dark 3 pair
            out_fields = get_output_fields_jit(ψ, φ, σ, λ, λ0, output_order)
            
            bright_field = out_fields[0]
            dark_fields = out_fields[1:]
            
            # Store intensities
            vector[p * 7:p * 7 + 6] = np.abs(dark_fields) ** 2
            vector[p * 7 + 6] = np.abs(bright_field) ** 2
            
        vector[-14:] = σ
        dataset[v] = vector
        
    return dataset

def get_random_dataset(size: int = 1000, λ: float = 1.65e-6, λ0: float = 1.65e-6) -> np.ndarray:
    """Build a random-point synthetic dataset.

    Args:
        size: Number of samples to generate.
        λ: Wavelength in meters.
        λ0: Reference wavelength in meters.

    Returns:
        np.ndarray of shape (size, vector_len).
    """
    nb_points = 100
    i_len = 7 + 14 # 7 outputs + 14 applied shifts
    o_len = 14 # 14 intrinsic errors (targets)
    vector_len = nb_points * i_len + o_len
    dataset = np.empty((size, vector_len))
    
    # Input fields
    ψ = np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j], dtype=complex)
    output_order = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    
    pv = 0
    for v in range(size):
        if (nv := (v * 100 // size)) > pv:
            print(f"{nv}%", end='\r')
            pv = nv
            
        # Intrinsic errors σ
        σ = np.random.uniform(0, 1, 14) * λ / 10
        
        vector = np.empty(vector_len)
        for p in range(nb_points):
            # Random applied shifts φ up to λ
            φ = np.random.uniform(0, λ, size=14)
            
            out_fields = get_output_fields_jit(ψ, φ, σ, λ, λ0, output_order)
            
            bright_field = out_fields[0]
            dark_fields = out_fields[1:]
            
            # Store: applied shifts + dark intensities + bright intensity
            vector[p * i_len:(p + 1) * i_len] = np.concatenate([φ, np.abs(dark_fields) ** 2, [np.abs(bright_field) ** 2]])
            
        vector[-14:] = σ
        dataset[v] = vector
        
    return dataset

def get_model(input_shape: int) -> Any:
    """Create a small dense Keras network for 14 output parameters.

    Args:
        input_shape: Input vector length.

    Returns:
        Compiled tf.keras.Model (Adam optimizer, MSE loss).
    """
    if tf is None:
        raise ImportError("TensorFlow is required for get_model")
        
    i = tf.keras.Input(shape=(input_shape,), name='Input')
    x = tf.keras.layers.Dense(128, activation='relu', name='Dense_1')(i)
    x = tf.keras.layers.Dense(64, activation='relu', name='Dense_2')(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='Dense_3')(x)
    o = tf.keras.layers.Dense(14, activation='relu', name='Output')(x)
    model = tf.keras.Model(inputs=i, outputs=o)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-05)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(model: Any, dataset: np.ndarray, plot: bool = True) -> Any:
    """Train the model and optionally plot the loss curves.

    Args:
        model: Compiled Keras model.
        dataset: Data with targets in the last 14 columns.
        plot: If True, plots loss and val_loss (log scale).

    Returns:
        tf.keras.callbacks.History
    """
    X = dataset[:, :-14]
    Y = dataset[:, -14:]
    print(dataset.shape, X.shape, Y.shape)
    history = model.fit(X, Y, epochs=10, batch_size=5, validation_split=0.2)
    
    if plot:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.yscale('log')
        plt.legend()
        plt.show()
        
    return history

def test_model(model: Any, dataset: Optional[np.ndarray] = None) -> None:
    """Plot ground-truth targets vs predictions for a quick visual check.

    Args:
        model: Trained Keras model.
        dataset: Test data; if None, generates 10 samples.

    Returns:
        None
    """
    if dataset is None:
        TEST_SET = get_dataset(size=10)
    else:
        TEST_SET = dataset
        
    X = TEST_SET[:, :-14]
    Y = TEST_SET[:, -14:]
    PREDICTIONS = model.predict(X)
    
    # print(X)
    # print(PREDICTIONS)
    
    plt.figure(figsize=(6, 6))
    for i in range(len(Y)):
        plt.scatter(Y[i], PREDICTIONS[i], alpha=0.5)
            
    plt.xlabel('Expectations')
    plt.ylabel('Predictions')
    
    # Add diagonal line
    min_val = min(np.min(Y), np.min(PREDICTIONS))
    max_val = max(np.max(Y), np.max(PREDICTIONS))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.show()