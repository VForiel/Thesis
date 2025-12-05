import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import deepcopy
import astropy.units as u
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path

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

def get_phase_map(context: Context, n_steps: int = 50, plot: bool = False) -> np.ndarray:
    """
    Perform a phase scan on each shifter to determine the output phases.
    
    Args:
        context (Context): The observation context.
        n_steps (int): Number of phase steps for the scan (0 to 2pi).
        plot (bool): Whether to plot the fits (for debugging).
        
    Returns:
        np.ndarray: A matrix of shape (N_shifters, N_outputs) containing the fitted phase offsets.
    """
    # Work on a copy to not disturb the original context
    ctx = deepcopy(context)
    
    # Ensure we are in monochromatic mode for calibration to avoid bandwidth smearing
    # or use the provided context settings. The user didn't specify, but calibration 
    # is usually done with a laser (monochromatic).
    # However, if we want to calibrate the "broadband" response, we should keep it as is.
    # Let's assume we use the context as provided, but maybe force monochromatic for speed if not specified.
    # For now, I'll stick to the context's observe method.
    
    n_shifters = len(ctx.interferometer.chip.φ)
    n_outputs = ctx.interferometer.chip.nb_raw_outputs
    
    # Result matrix
    phase_map = np.zeros((n_shifters, n_outputs))
    
    # Scan range: 0 to 2pi (in terms of phase shift)
    # Chip.φ is likely an OPD (length) or Phase (rad). 
    # In phise/classes/context.py: self.interferometer.chip.φ[i-1] += Δφ where Δφ = λ/4.
    # This implies φ is a length (OPD).
    # So we need to scan OPD from 0 to λ.
    wavelength = ctx.interferometer.λ
    opd_range = np.linspace(0, wavelength.to(u.m).value, n_steps) * u.m
    phase_range_rad = np.linspace(0, 2*np.pi, n_steps) # For plotting/fitting x-axis
    
    # Store original phases
    original_phases = ctx.interferometer.chip.φ.copy()
    
    if plot:
        fig, axs = plt.subplots(n_shifters, 1, figsize=(10, 4*n_shifters), constrained_layout=True)
        if n_shifters == 1: axs = [axs]
    
    for i in range(n_shifters):
        fluxes = []
        
        # Reset all phases to original before scanning this shifter?
        # Or keep others at original? Yes, keep others at original.
        ctx.interferometer.chip.φ = original_phases.copy()
        
        # Scan
        for opd in opd_range:
            # Set phase for shifter i
            # Note: φ is likely 0-indexed in the array, but 1-indexed in labels.
            # We iterate 0 to N-1.
            # We add the scan offset to the original value? 
            # The user says "scan de phase de 0 à 2pi". Usually this means replacing the value.
            # But if there is a bias, we might want to scan around it.
            # Let's assume we replace the value or add to 0.
            # If we want to characterize the *current* state, we should probably scan *around* the current value
            # or just sweep the full range.
            # Let's sweep the full range [0, lambda] replacing the current value.
            ctx.interferometer.chip.φ[i] = opd
            
            # Observe
            # We use observe_monochromatic for speed and clarity if possible, 
            # but observe() is safer if the context is broadband.
            # We assume upstream_pistons=0 (no atmospheric turbulence) for calibration.
            outs = ctx.observe(upstream_pistons=np.zeros(len(ctx.interferometer.telescopes))*u.m)
            fluxes.append(outs)
            
        fluxes = np.array(fluxes) # (n_steps, n_outputs)
        
        # Fit for each output
        for j in range(n_outputs):
            y_data = fluxes[:, j]
            
            # Check if this output is modulated
            amplitude = np.ptp(y_data)
            if amplitude < 1e-9: # Threshold for "no signal" or "noise only"
                phase_map[i, j] = 0 # Or NaN? 0 is safer for NN input
                continue
                
            # Initial guess
            # A * sin(B*x + C) + E
            # x is phase_range_rad (0 to 2pi)
            # B should be 1
            A_guess = amplitude / 2
            E_guess = np.mean(y_data)
            B_guess = 1.0
            C_guess = 0.0
            D_guess = 0.0
            F_guess = 0.0
            
            p0 = [A_guess, B_guess, C_guess, D_guess, E_guess, F_guess]
            
            try:
                # Constrain B to be close to 1?
                # The user said "même que dans Kbench", Kbench doesn't constrain B much but checks period.
                popt, _ = curve_fit(sine_func, phase_range_rad, y_data, p0=p0, maxfev=2000)
                
                # Extract C (phase offset)
                # We want the phase of the oscillation.
                # Model: sin(x + C). 
                # We store C modulo 2pi.
                phase_offset = popt[2] % (2*np.pi)
                phase_map[i, j] = phase_offset
                
                if plot:
                    axs[i].plot(phase_range_rad, y_data, '.', label=f'Out {j} Data')
                    axs[i].plot(phase_range_rad, sine_func(phase_range_rad, *popt), '-', label=f'Out {j} Fit')
            except Exception:
                phase_map[i, j] = 0.0 # Fit failed
        
        if plot:
            axs[i].set_title(f"Shifter {i} Scan")
            axs[i].legend()

    # Restore context
    ctx.interferometer.chip.φ = original_phases
    
    return phase_map

def generate_dataset(context: Context, n_samples: int=100, n_steps: int=20):
    """
    Generate a dataset for training the neural network.
    
    Args:
        context (Context): Base context.
        n_samples (int): Number of samples.
        n_steps (int): Number of steps for the phase scan (quality of input map).
        
    Returns:
        tuple: (X, y) where X is (n_samples, n_features) and y is (n_samples, n_targets).
    """
    # Define cache path
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
    
    n_shifters = len(context.interferometer.chip.φ)
    n_outputs = context.interferometer.chip.nb_raw_outputs
    
    X = [] # Inputs: Flattened phase maps
    y = [] # Targets: Correction phases
    
    wavelength = context.interferometer.λ
    
    print(f"Generating dataset with {n_samples} samples...")
    for _ in tqdm(range(n_samples)):
        # Generate random perturbation (phases in radians)
        # We want to cover the full 0-2pi range to be robust?
        # Or small perturbations? The user said "plein de contexts perturbés".
        # Let's pick random phases in [0, 2pi].
        random_phases_rad = np.random.uniform(0, 2*np.pi, n_shifters)
        
        # Convert to OPD
        random_opd = (random_phases_rad / (2*np.pi)) * wavelength.to(u.m).value * u.m
        
        # Apply to context
        # We assume the "base" context has 0 phases or we just overwrite them.
        # Let's overwrite.
        context.interferometer.chip.φ = random_opd
        
        # Run calibration
        # We use a small number of steps to be faster? 
        # User said "scan de 0 à 2pi". 
        # If we want to be fast, maybe 20 steps is enough.
        phase_map = get_phase_map(context, n_steps=n_steps, plot=False)
        
        # Input vector: Flattened phase map
        X.append(phase_map.flatten())
        
        # Target vector: The ideal phases to correct this state.
        # If current state is phi, we want to apply -phi (modulo 2pi) to get back to 0.
        # Or maybe we want to reach a specific target?
        # Assuming we want to zero the phases (constructive interference or specific state).
        # The user said: "phases optimisées (que tu peux obtenir analytiquement en faisant φ=[-σ]%2pi)"
        # So target is [-random_phases_rad] % 2pi.
        target = (-random_phases_rad) % (2*np.pi)
        y.append(target)
        
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
    
    print(f"Generating test dataset with {len(test_phases)*n_repeats} samples...")
    
    for target_phase in tqdm(test_phases):
        for _ in range(n_repeats):
            # Ensure positive OPD by taking modulo 2pi
            phase_val = (-target_phase) % (2*np.pi)
            current_opd = (phase_val / (2*np.pi)) * wavelength.to(u.m).value * u.m
            
            context.interferometer.chip.φ = np.full(n_shifters, current_opd.value) * current_opd.unit
            
            # Get input map
            phase_map = get_phase_map(context, n_steps=n_steps, plot=False)
            
            X.append(phase_map.flatten())
            y.append([target_phase] * n_shifters)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Saving test dataset to {file_path}...")
    np.savez(file_path, X=X, y=y)
    
    return X, y

def preprocess_data(X, y=None):
    """
    Convert phase data (radians) to sin/cos components to avoid phase wrapping issues.
    
    Args:
        X (np.ndarray): Input phases (N_samples, N_features).
        y (np.ndarray, optional): Target phases (N_samples, N_targets).
        
    Returns:
        tuple: (X_processed, y_processed)
    """
    # Convert inputs: [cos(X), sin(X)]
    X_cos = np.cos(X)
    X_sin = np.sin(X)
    X_proc = np.concatenate([X_cos, X_sin], axis=1)
    
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

class CosineLoss(nn.Module):
    """
    Loss function that maximizes the cosine similarity between predicted and target vectors.
    Since we predict (cos, sin) pairs, maximizing dot product is equivalent to minimizing angular error.
    """
    def __init__(self):
        super(CosineLoss, self).__init__()
        
    def forward(self, pred, target):
        # pred and target are (batch, 2*n_targets)
        # We reshape to (batch, n_targets, 2) to handle pairs
        n_targets = pred.shape[1] // 2
        
        pred_pairs = pred.view(-1, n_targets, 2)
        target_pairs = target.view(-1, n_targets, 2)
        
        # Normalize predictions to ensure they are on the unit circle
        # (Targets are already unit vectors by construction)
        pred_norm = torch.nn.functional.normalize(pred_pairs, p=2, dim=2)
        
        # Cosine similarity: dot product
        # sum over the 2 components (cos*cos + sin*sin)
        cosine_sim = torch.sum(pred_norm * target_pairs, dim=2)
        
        # Loss = 1 - mean(cosine_similarity)
        # We want cosine_sim to be 1.
        loss = 1.0 - torch.mean(cosine_sim)
        
        return loss

class CalibrationNet(nn.Module):
    """
    A Multi-Layer Perceptron implemented in PyTorch.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super(CalibrationNet, self).__init__()
        # Simplified architecture for small dataset
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

def train_calibration_model(X, y, hidden_size=128, epochs=500, learning_rate=0.001, batch_size=32, dropout_prob=0.1):
    """
    Create and train the neural network model using PyTorch.
    
    Args:
        X (np.ndarray): Input data (N_samples, N_features).
        y (np.ndarray): Target data (N_samples, N_targets).
        hidden_size (int): Number of neurons in the hidden layer.
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
    
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    # Initialize model, loss function, and optimizer
    model = CalibrationNet(input_size, hidden_size, output_size, dropout_prob=dropout_prob)
    
    # Use Cosine Loss for phase regression
    criterion = CosineLoss()
    
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
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    print("Training complete.")
    
    return model, loss_history

def plot_sensitivity_comparison(model, context, n_observations=50):
    """
    Compare null depth sensitivity to atmospheric noise for:
    1. Ideal case (perfect cophasing)
    2. Perturbed case (random errors)
    3. Calibrated case (AI correction)
    """
    print("\n--- Running Sensitivity Analysis ---")
    
    # Define range of atmospheric noise (Γ)
    gammas = np.linspace(0, 300, 10) * u.nm
    
    null_depths_ideal = []
    null_depths_perturbed = []
    null_depths_calibrated = []
    
    # Store original state
    original_gamma = context.Γ
    original_phases = deepcopy(context.interferometer.chip.φ)
    
    wavelength = context.interferometer.λ
    n_shifters = len(context.interferometer.chip.φ)
    
    for gamma in tqdm(gammas, desc="Scanning Gamma"):
        context.Γ = gamma
        
        fluxes_ideal = []
        fluxes_perturbed = []
        fluxes_calibrated = []
        
        for _ in range(n_observations):
            # --- Ideal ---
            context.interferometer.chip.φ = np.zeros(n_shifters) * u.m
            # Force recomputation of atmospheric phase screens
            context._update_pf() 
            outs = context.observe()
            # Null depth = sum(Dark) / Bright. Dark are indices 1-6. Bright is 0.
            # Avoid division by zero if bright is 0 (unlikely)
            nd = np.sum(outs[1:]) / (outs[0] + 1e-10)
            fluxes_ideal.append(nd)
            
            # --- Perturbed ---
            # Apply a static instrumental error (e.g. 50nm rms)
            random_phases = np.random.normal(0, 0.5, n_shifters) # radians
            opd_err = (random_phases / (2*np.pi)) * wavelength.to(u.m).value * u.m
            # Ensure positive OPD by adding a large offset if needed, or just take modulo if the chip supports it.
            # But here we are simulating errors. The chip class seems to enforce positive OPD.
            # Let's assume we can just add a base OPD or take modulo.
            # Actually, let's just make sure it's positive by adding multiple of wavelength if negative.
            opd_err = (opd_err % wavelength)
            
            context.interferometer.chip.φ = opd_err
            context._update_pf()
            outs = context.observe()
            nd = np.sum(outs[1:]) / (outs[0] + 1e-10)
            fluxes_perturbed.append(nd)
            
            # --- Calibrated ---
            # 1. Get phase map (scan)
            phase_map = get_phase_map(context, n_steps=20, plot=False)
            # 2. Predict correction
            X_in_raw = phase_map.flatten().reshape(1, -1)
            X_in_proc, _ = preprocess_data(X_in_raw)
            X_in = torch.tensor(X_in_proc, dtype=torch.float32)
            
            pred_proc = model(X_in).detach().numpy()
            pred_correction_rad = recover_phases(pred_proc).flatten()
            
            # 3. Apply correction
            correction_opd = (pred_correction_rad / (2*np.pi)) * wavelength.to(u.m).value * u.m
            
            # Apply correction to the *current* state (which is opd_err)
            context.interferometer.chip.φ = opd_err + correction_opd
            context._update_pf()
            outs = context.observe()
            nd = np.sum(outs[1:]) / (outs[0] + 1e-10)
            fluxes_calibrated.append(nd)
            
        null_depths_ideal.append(np.mean(fluxes_ideal))
        null_depths_perturbed.append(np.mean(fluxes_perturbed))
        null_depths_calibrated.append(np.mean(fluxes_calibrated))
        
    # Restore context
    context.Γ = original_gamma
    context.interferometer.chip.φ = original_phases
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gammas.value, null_depths_ideal, 'g-o', label='Ideal (No Static Error)')
    plt.plot(gammas.value, null_depths_perturbed, 'r-x', label='Perturbed (Static Error)')
    plt.plot(gammas.value, null_depths_calibrated, 'b-^', label='AI Calibrated')
    
    plt.xlabel(f"Atmospheric Noise Γ ({gammas.unit})")
    plt.ylabel("Null Depth (Dark/Bright)")
    plt.title("Sensitivity Analysis: Calibration Performance vs Atmosphere")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    # --- 1. Setup Context ---
    print("--- Setting up Simulation Context ---")
    
    # Use the factory method to get a valid context
    ctx = Context.get_VLTI()
    
    # Adjust parameters for this specific simulation
    ctx.h = 0 * u.hourangle
    ctx.Δh = 1 * u.hourangle
    ctx.Γ = 10 * u.nm # Small cophasing error
    
    # Adjust target to be a simple star without companions for calibration
    # Vega is magnitude 0 roughly. We want magnitude 5 (100x fainter).
    ctx.target.f *= 0.01 
    ctx.target.companions = []
    
    n_shifters = 14
    
    # --- 2. Generate Dataset ---
    print("\n--- Generating Training Dataset ---")
    # Increased size and quality as requested
    X_train_raw, y_train_raw = generate_dataset(ctx, n_samples=2000, n_steps=20)
    
    # Preprocess data (Phase -> Sin/Cos)
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
    
    # --- 3. Train Model ---
    print("\n--- Training Model ---")
    # Tuned hyperparameters for small dataset:
    # - Smaller hidden size to avoid overfitting
    # - Cosine Loss
    model, loss_history = train_calibration_model(X_train, y_train, hidden_size=128, epochs=1000, learning_rate=0.001, dropout_prob=0.1)
    
    # --- 4. Test and Plot ---
    print("\n--- Testing and Plotting ---")
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
    
    # Generate Test Data for Violin Plot
    # We want discrete expected phases
    test_phases = np.linspace(0, 2*np.pi, 8)
    n_repeats = 5 # Number of repeats per phase to show dispersion
    
    print("Generating test samples...")
    X_test_raw, y_test_raw = generate_test_dataset(ctx, test_phases, n_repeats=n_repeats, n_steps=20)
    
    # Preprocess input
    X_test_proc, _ = preprocess_data(X_test_raw)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)
    
    # Enable dropout for inference (Monte Carlo Dropout)
    model.train() 
    
    # Predict
    with torch.no_grad():
        pred_proc = model(X_test_tensor).detach().numpy()
        
    # Recover phases
    predicted_phases_all = recover_phases(pred_proc).flatten()
    expected_phases_all = y_test_raw.flatten()

    # Violin Plot
    plt.figure(figsize=(12, 6))
    
    # Organize data for violin plot
    data_to_plot = []
    labels = []
    
    expected_phases_all = np.array(expected_phases_all)
    predicted_phases_all = np.array(predicted_phases_all)
    
    for tp in test_phases:
        # We need to handle phase wrapping in comparison too
        # But here we just plot raw predicted values vs expected
        mask = np.isclose(expected_phases_all, tp)
        preds = predicted_phases_all[mask]
        
        # Unwrap for plotting if needed, but here we expect [0, 2pi]
        # If prediction is near 0 and target is 2pi, it looks bad but is good.
        # Let's shift predictions to be close to target for visualization
        preds_shifted = preds.copy()
        # Simple unwrap logic for visualization: if diff > pi, shift by 2pi
        diff = preds - tp
        preds_shifted[diff > np.pi] -= 2*np.pi
        preds_shifted[diff < -np.pi] += 2*np.pi
        
        data_to_plot.append(preds_shifted)
        labels.append(f"{tp:.2f}")
        
    plt.violinplot(data_to_plot, positions=range(len(test_phases)), showmeans=True)
    plt.xticks(range(len(test_phases)), labels)
    plt.xlabel("Expected Phase (rad)")
    plt.ylabel("Predicted Phase (rad)")
    plt.title("Model Performance: Predicted vs Expected Phase (Sin/Cos Space)")
    plt.grid(True, axis='y')
    
    plt.plot(range(len(test_phases)), test_phases, 'r--', label='Ideal', alpha=0.5)
    plt.legend()
    plt.show()
    
    # --- 5. Sensitivity Analysis Plot ---
    plot_sensitivity_comparison(model, ctx, n_observations=50)
