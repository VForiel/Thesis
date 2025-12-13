import sys
from pathlib import Path
# Add src to path to allow running without installing
sys.path.append(str(Path.cwd() / "src"))

import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
import astropy.units as u
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations

from phise.classes.context import Context
from phise.classes.telescope import Telescope
from phise.classes.interferometer import Interferometer
from phise.classes.camera import Camera
from phise.classes.target import Target
from phise.classes.archs.superkn import SuperKN

def sine_func(x, A, B, C, D, E, F):
    """
    Model function for fitting: (A + F*x) * sin(B*x + C) + D*x + E
    In simulation, we expect F=0, D=0, B=1.
    """
    return (A + F * x) * np.sin(B * x + C) + D * x + E

def sine_func_simple(x, A, C, E):
    """
    Simplified model function: A * sin(x + C) + E
    Assuming B=1, D=0, F=0.
    """
    return A * np.sin(x + C) + E

def get_phase_visibility_map_full(context: Context, n_steps: int = 50, base_phases: u.Quantity = None) -> tuple:
    """
    Perform a comprehensive phase scan for all input combinations.
    Now supports explicit base_phases to avoid modifying context.
    """
    # NO DEEPCOPY needed if we don't mutate context
    ctx = context 
    
    n_shifters = len(ctx.interferometer.chip.φ)
    n_outputs = ctx.interferometer.chip.nb_raw_outputs
    n_inputs = len(ctx.interferometer.telescopes)
    
    # ... (combinations logic same as before) ...
    # Generate all input combinations
    all_input_indices = list(range(n_inputs))
    input_combinations = []
    
    # 1-input: [0], [1], [2], [3]
    for i in all_input_indices:
        input_combinations.append([i])
    
    # 2-inputs
    for combo in combinations(all_input_indices, 2):
        input_combinations.append(list(combo))
    
    # 3-inputs
    for combo in combinations(all_input_indices, 3):
        input_combinations.append(list(combo))
    
    # 4-inputs
    input_combinations.append(all_input_indices)
    
    all_phases = []
    all_visibilities = []
    
    wavelength = ctx.interferometer.λ
    opd_range = np.linspace(0, wavelength.to(u.m).value, n_steps) * u.m
    phase_range_rad = np.linspace(0, 2*np.pi, n_steps)
    
    # Use provided base_phases or default from context
    if base_phases is not None:
        original_phases = base_phases
    else:
        original_phases = ctx.interferometer.chip.φ
    
    photon_flux_per_tel = ctx.pf
    
    for active_inputs in input_combinations:
        base_input_fields = np.zeros(n_inputs, dtype=complex)
        for idx in active_inputs:
            base_input_fields[idx] = np.sqrt(photon_flux_per_tel[idx])
        
        n_batch = n_shifters * n_steps
        phi_batch = np.zeros((n_batch, n_shifters))
        
        base_phases_val = original_phases.to(u.m).value
        
        idx = 0
        for shifter_idx in range(n_shifters):
            for opd in opd_range:
                current_phases = base_phases_val.copy()
                current_phases[shifter_idx] += opd.to(u.m).value
                phi_batch[idx] = current_phases
                idx += 1
                
        input_fields_batch = np.tile(base_input_fields, (n_batch, 1))
        sigma_batch = np.tile(ctx.interferometer.chip.σ.to(u.m).value, (n_batch, 1)) * u.m
        
        output_fields_batch = ctx.interferometer.chip.get_output_fields(
            ψ=input_fields_batch,
            λ=wavelength,
            σ=sigma_batch,
            φ=phi_batch * u.m
        )
        
        fluxes_batch = np.abs(output_fields_batch)**2
        
        # ... (FFT logic same as previous step) ...
        # Copied from previous edit
        fluxes = fluxes_batch.reshape(n_shifters, n_steps, n_outputs)
        fluxes_t = fluxes.transpose(0, 2, 1)
        fluxes_flat = fluxes_t.reshape(-1, n_steps)
        fluxes_in = fluxes_flat[:, :-1]
        N = n_steps - 1
        Y = np.fft.rfft(fluxes_in, axis=1)
        Y1 = Y[:, 1]
        phases = (np.angle(Y1) + np.pi/2) % (2*np.pi)
        metrics_vis = np.ptp(fluxes_flat, axis=1)
        peaks = np.max(fluxes_flat, axis=1)
        thresholds = np.maximum(1.0, peaks * 0.001)
        mask = metrics_vis < thresholds
        phases[mask] = 0.0
        metrics_vis[mask] = 0.0
        all_phases.extend(phases.tolist())
        all_visibilities.extend(metrics_vis.tolist())
    
    # No context restoration needed if we didn't mutate it
    
    phases_vector = np.array(all_phases)
    visibilities_vector = np.array(all_visibilities)
    
    return phases_vector, visibilities_vector

def generate_dataset(context: Context, n_samples: int=100, n_steps: int=20, max_workers: int=None):
    """
    Generate a dataset for training the neural network.
    """
    import os
    import copy
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        import psutil
    except ImportError:
        psutil = None
    
    cache_dir = Path("generated/dataset/phase_maps")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename based on parameters
    # We include n_samples, n_steps, and Gamma (atmospheric noise)
    gamma_val = int(context.Γ.value) if hasattr(context.Γ, 'value') else 0
    filename = f"dataset_ns{n_samples}_nst{n_steps}_gamma{gamma_val}.npz"
    file_path = cache_dir / filename
    
    if file_path.exists():
        print(f"Loading cached dataset from {file_path}...")
        data = np.load(file_path)
        return data['X'], data['y']
        
    # Check if a larger dataset exists and subsample from it
    # We look for files matching the pattern but with different ns
    for existing_file in cache_dir.glob(f"dataset_ns*_nst{n_steps}_gamma{gamma_val}.npz"):
        try:
            # Extract n_samples from filename
            parts = existing_file.stem.split('_')
            for p in parts:
                if p.startswith('ns'):
                    ns_existing = int(p[2:])
                    if ns_existing > n_samples:
                        print(f"Found larger dataset {existing_file.name} ({ns_existing} samples). Subsampling {n_samples} samples...")
                        data = np.load(existing_file)
                        X_full, y_full = data['X'], data['y']
                        
                        # Random subsample
                        indices = np.random.choice(ns_existing, n_samples, replace=False)
                        return X_full[indices], y_full[indices]
        except Exception as e:
            print(f"Failed to read/subsample {existing_file}: {e}")
            continue

    if file_path.exists(): # Double check just in case logic above fell through
         pass
    
    n_shifters = len(context.interferometer.chip.φ)
    n_outputs = context.interferometer.chip.nb_raw_outputs
    
    X = []
    y = []
    
    wavelength = context.interferometer.λ
    
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) if psutil else None
    print(f"Generating dataset with {n_samples} samples... (workers={max_workers}, physical={physical_cores})")

    def _generate_one_sample(_seed: int=None):
        # NO DEEPCOPY: Use shared context, pass dynamic phases
        # Generate random perturbation
        rng = np.random.default_rng(_seed)
        random_phases_rad = rng.uniform(0, 2*np.pi, n_shifters)
        # Convert to OPD
        random_opd = (random_phases_rad / (2*np.pi)) * wavelength.to(u.m).value * u.m
        
        # Compute phase and visibility maps using explicit base_phases
        # Pass context as is (it won't be modified)
        phases_vector, visibilities_vector = get_phase_visibility_map_full(context, n_steps=n_steps, base_phases=random_opd)
        
        x_row = np.concatenate([phases_vector, visibilities_vector])
        y_row = (-random_phases_rad) % (2*np.pi)
        return x_row, y_row

    # Parallel execution
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(n_samples):
            futures.append(ex.submit(_generate_one_sample, i))
        for f in tqdm(as_completed(futures), total=len(futures)):
            x_row, y_row = f.result()
            X.append(x_row)
            y.append(y_row)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Saving dataset to {file_path}...")
    np.savez(file_path, X=X, y=y)
        
    return X, y

def generate_test_dataset(context: Context, test_phases: np.ndarray, n_repeats: int=5, n_steps: int=20):
    """
    Generate a test dataset for evaluating the neural network.
    
    Args:
        context (Context): Base context.
        test_phases (np.ndarray): Array of target phases to test.
        n_repeats (int): Number of repeats per phase.
        n_steps (int): Number of steps for the phase scan.
        
    Returns:
        tuple: (X, y) where X is (n_samples, n_features) and y is (n_samples, n_targets).
               Here y contains the expected phases for each shifter.
    """
    # Define cache path
    cache_dir = Path("generated/dataset/test_phase_maps")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename based on parameters
    gamma_val = int(context.Γ.value) if hasattr(context.Γ, 'value') else 0
    n_phases = len(test_phases)
    filename = f"test_dataset_np{n_phases}_nr{n_repeats}_nst{n_steps}_gamma{gamma_val}.npz"
    file_path = cache_dir / filename
    
    if file_path.exists():
        print(f"Loading cached test dataset from {file_path}...")
        data = np.load(file_path)
        return data['X'], data['y']
    
    n_shifters = len(context.interferometer.chip.φ)
    wavelength = context.interferometer.λ
    
    X = []
    y = []
    
    # Generate random test samples (same distribution as training)
    # This ensures we aren't testing on OOD data (uniform phases)
    print(f"Generating random test dataset with {len(test_phases)*n_repeats} samples...")
    
    # We ignore test_phases values and just use the count
    n_total_test = len(test_phases) * n_repeats
    
    local_ctx = copy.deepcopy(context)
    
    for i in tqdm(range(n_total_test)):
        # Generate random perturbation
        # Use a deterministic seed offset for reproducibility of test set
        rng = np.random.default_rng(1000000 + i) 
        random_phases_rad = rng.uniform(0, 2*np.pi, n_shifters)
        
        # Calculate OPD
        random_opd = (random_phases_rad / (2*np.pi)) * wavelength.to(u.m).value * u.m
        
        # Get input map
        # pass context as is
        phases_vector, visibilities_vector = get_phase_visibility_map_full(context, n_steps=n_steps, base_phases=random_opd)
        
        # Concatenate
        input_vector = np.concatenate([phases_vector, visibilities_vector])
        X.append(input_vector)
        
        # Target: [-sigma] % 2pi
        # Same as training: we want to predict the correction
        y_row = (-random_phases_rad) % (2*np.pi)
        y.append(y_row)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Saving test dataset to {file_path}...")
    np.savez(file_path, X=X, y=y)
    
    return X, y

def preprocess_data(X, n_shifters, n_outputs, y=None):
    """
    Convert phase/visibility data to sin/cos components.
    
    Args:
        X (np.ndarray): Input data (N_samples, 1680) where 1680 = 840 phases + 840 visibilities.
        n_shifters (int): Number of shifters (14).
        n_outputs (int): Number of outputs (4).
        y (np.ndarray, optional): Target phases (N_samples, 14).
        
    Returns:
        tuple: (X_processed, y_processed)
        X_processed: (N_samples, 2520) - phases converted to sin/cos, visibilities normalized
    """
    N = X.shape[0]
    
    # Split phases and visibilities
    # X shape: (N, 2*M) where first M are phases, last M are visibilities
    n_features = X.shape[1]
    half_size = n_features // 2
    
    phases = X[:, :half_size]  # (N, half_size)
    visibilities = X[:, half_size:]  # (N, half_size)
    
    # Convert phases to sin/cos
    phases_sin = np.sin(phases)
    phases_cos = np.cos(phases)
    
    # Normalize visibilities (simple min-max normalization per sample)
    # To avoid division by zero, add small epsilon
    vis_max = visibilities.max(axis=1, keepdims=True) + 1e-9
    visibilities_norm = visibilities / vis_max
    
    # Concatenate: [sin(phases), cos(phases), visibilities_norm]
    # Total: 1470 + 1470 + 1470 = 4410 features
    X_proc = np.concatenate([phases_sin, phases_cos, visibilities_norm], axis=1)
    
    y_proc = None
    if y is not None:
        # Convert targets: [cos(y), sin(y)]
        y_cos = np.cos(y)
        y_sin = np.sin(y)
        y_proc = np.concatenate([y_cos, y_sin], axis=1)
        
    return X_proc, y_proc

def recover_phases(y_proc):
    """
    Recover phases from sin/cos components.
    
    Args:
        y_proc (np.ndarray): (N_samples, 2*N_targets) [cos, sin]
        
    Returns:
        np.ndarray: Phases in [0, 2pi]
    """
    n_targets = y_proc.shape[1] // 2
    y_cos = y_proc[:, :n_targets]
    y_sin = y_proc[:, n_targets:]
    
    phases = np.arctan2(y_sin, y_cos)
    return phases % (2*np.pi)

class UnitCircleMSELoss(nn.Module):
    """
    MSE Loss with regularization to enforce unit circle constraint (sin^2 + cos^2 = 1).
    """
    def __init__(self, lambda_reg=0.1):
        super(UnitCircleMSELoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # MSE Loss
        loss_mse = self.mse(pred, target)
        
        # Unit Circle Regularization: (sin^2 + cos^2 - 1)^2
        # pred is (Batch, 2*N_targets)
        n_targets = pred.shape[1] // 2
        pred_cos = pred[:, :n_targets]
        pred_sin = pred[:, n_targets:]
        norm_sq = pred_cos**2 + pred_sin**2
        loss_reg = torch.mean((norm_sq - 1.0)**2)
        
        return loss_mse + self.lambda_reg * loss_reg

class CalibrationNet(nn.Module):
    """
    A Fully Connected Neural Network (MLP) implemented in PyTorch.
    Processes concatenated phase/visibility vectors.
    """
    def __init__(self, input_size, output_size, dropout_prob=0.05):
        super(CalibrationNet, self).__init__()
        
        # Input: (batch, 4410) = sin/cos phases + visibilities
        # Output: (batch, 28) = sin/cos for 14 shifter target phases
        
        # Scaling up for 100k dataset
        # Width: 2048 -> 1024 -> 512 -> 256
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Dropout(dropout_prob),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout_prob),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout_prob),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_prob),
            
            nn.Linear(256, output_size * 2)  # sin/cos for each target
        )
        
    def forward(self, x):
        return self.network(x)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

def train_calibration_model(X, y, n_shifters, n_outputs, epochs=500, learning_rate=0.001, batch_size=32, dropout_prob=0.05):
    """
    Create and train the neural network model using PyTorch.
    
    Args:
        X (np.ndarray): Input data (N_samples, 4410) with sin/cos phases + normalized visibilities.
        y (np.ndarray): Target data (N_samples, 28) - sin/cos for each of 14 target phases.
        n_shifters (int): Number of shifters (14).
        n_outputs (int): Number of outputs (4).
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dropout_prob (float): Dropout probability.
        
    Returns:
        CalibrationNet: Trained PyTorch model.
        list: Loss history.
    """
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    output_size = y.shape[1] // 2  # Should be 14 (number of shifters)
    input_size = X.shape[1]  # Should be 4410 after preprocessing
    
    # Initialize model, loss function, and optimizer
    model = CalibrationNet(input_size, output_size, dropout_prob=dropout_prob)
    
    # Use UnitCircleMSELoss to enforce sin^2+cos^2=1
    criterion = UnitCircleMSELoss(lambda_reg=0.5)
    
    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    print("Training neural network with PyTorch...")
    loss_history = []
    
    for epoch in range(epochs):
        model.train() # Ensure dropout is active during training
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    print("Training complete.")
    
    return model, loss_history



import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train calibration network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout probability')
    parser.add_argument('--samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--steps', type=int, default=20, help='Number of phase scan steps')
    parser.add_argument('--note', type=str, default="", help='Note for the log')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gamma', type=float, default=9.0, help='Cophasing error (Gamma) in nm')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--generate-only', action='store_true', help='Only generate dataset and exit')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # --- 1. Setup Context ---
    print("--- Setting up Simulation Context ---")
    
    # Use the factory method to get a valid context
    ctx = Context.get_VLTI()
    
    # Adjust parameters for this specific simulation
    ctx.h = 0 * u.hourangle
    ctx.Δh = 1 * u.hourangle
    ctx.Γ = args.gamma * u.nm # Small cophasing error
    
    # Adjust target to be a simple star without companions for calibration
    # Vega is magnitude 0 roughly. We want magnitude 5 (100x fainter).
    ctx.target.f *= 0.01 
    ctx.target.companions = []
    
    n_shifters = len(ctx.interferometer.chip.φ)
    n_outputs = ctx.interferometer.chip.nb_raw_outputs
    
    # --- 2. Generate Dataset ---
    print("\n--- Generating Training Dataset ---")
    # Increased size and quality as requested
    X_train_raw, y_train_raw = generate_dataset(ctx, n_samples=args.samples, n_steps=args.steps)

    if args.generate_only:
        print(f"Dataset generated with {args.samples} samples. Exiting as requested.")
        sys.exit(0)
    
    # Preprocess data (Phase -> Sin/Cos)
    X_train, y_train = preprocess_data(X_train_raw, n_shifters, n_outputs, y_train_raw)
    
    # --- 3. Train Model ---
    print("\n--- Training Model ---")
    # Tuned hyperparameters for small dataset:
    # - Smaller hidden size to avoid overfitting
    # - Cosine Loss
    model, loss_history = train_calibration_model(X_train, y_train, n_shifters, n_outputs, epochs=args.epochs, learning_rate=args.lr, dropout_prob=args.dropout, batch_size=args.batch_size)
    
    # --- 4. Test and Plot ---
    print("\n--- Testing and Plotting ---")
    
    # Generate Test Data for Evaluation
    # We want discrete expected phases
    test_phases = np.linspace(0, 2*np.pi, 8)
    n_repeats = 5 # Number of repeats per phase to show dispersion
    
    print("Generating test samples...")
    X_test_raw, y_test_raw = generate_test_dataset(ctx, test_phases, n_repeats=n_repeats, n_steps=args.steps)
    
    # Preprocess input
    X_test_proc, _ = preprocess_data(X_test_raw, n_shifters, n_outputs)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)
    
    # Enable dropout for inference (Monte Carlo Dropout)
    model.train() 
    
    # --- Evaluate on Train Set ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        pred_train = model(X_train_tensor).detach().numpy()
    
    pred_phases_train = recover_phases(pred_train).flatten()
    expected_phases_train = y_train_raw.flatten()
    diff_train = np.abs(pred_phases_train - expected_phases_train)
    diff_train = np.minimum(diff_train, 2*np.pi - diff_train)
    mse_train = np.mean(diff_train**2)
    print(f"Train MSE: {mse_train:.6f}")

    # --- Evaluate on Test Set ---
    # Predict
    with torch.no_grad():
        pred_proc = model(X_test_tensor).detach().numpy()
        
    # Recover phases
    predicted_phases_all = recover_phases(pred_proc).flatten()
    expected_phases_all = y_test_raw.flatten()

    # Calculate MSE
    # We need to handle phase wrapping for MSE calculation
    # Error = min(|diff|, 2pi - |diff|)
    diff = np.abs(predicted_phases_all - expected_phases_all)
    diff = np.minimum(diff, 2*np.pi - diff)
    mse = np.mean(diff**2)
    
    print(f"Sample Prediction (first 5): {predicted_phases_all[:5]}")
    print(f"Sample Target (first 5):     {expected_phases_all[:5]}")
    
    final_loss = loss_history[-1] if loss_history else 0.0
    duration = time.time() - start_time
    
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Duration: {duration:.2f}s")
    
    # Log results
    log_entry = f"| {int(time.time())} | Epochs={args.epochs}, LR={args.lr}, Drop={args.dropout}, Samples={args.samples} | {final_loss:.6f} | {mse:.6f} (Train: {mse_train:.6f}) | {duration:.2f}s | {args.note} |\n"
    
    with open("optimization_log.md", "a") as f:
        f.write(log_entry)

    if not args.no_plot:
        # Plot Loss
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True)
        
        # Plot Scatter: Expected vs Predicted
        plt.subplot(1, 2, 2)
        plt.scatter(expected_phases_all, predicted_phases_all, alpha=0.1, s=1)
        plt.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', label='Ideal')
        plt.xlabel('Expected Phase (rad)')
        plt.ylabel('Predicted Phase (rad)')
        plt.title(f'Prediction vs Target (MSE={mse:.4f})')
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 2*np.pi)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

