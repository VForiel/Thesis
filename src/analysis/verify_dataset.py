import numpy as np
from pathlib import Path
import sys

# Add src to path just in case
sys.path.append(str(Path.cwd() / "src"))

# Load the dataset
file_path = Path('generated/dataset/phase_maps/dataset_ns100000_nst20_gamma9.npz')
if not file_path.exists():
    print(f"Dataset not found: {file_path}")
    sys.exit(1)

print(f"Loading {file_path}...")
data = np.load(file_path)
X = data['X']
y = data['y']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("X stats:")
print(f"  Min: {X.min()}")
print(f"  Max: {X.max()}")
print(f"  Mean: {X.mean()}")
print(f"  NaNs: {np.isnan(X).sum()}")

print("y stats:")
print(f"  Min: {y.min()}")
print(f"  Max: {y.max()}")
print(f"  Mean: {y.mean()}")
print(f"  NaNs: {np.isnan(y).sum()}")

# Check if y is within [0, 2pi]
print(f"y in [0, 2pi]: {np.all((y >= 0) & (y <= 2*np.pi + 1e-5))}")

# Check Y stats
print("y stats:")
print(f"  Min: {np.min(y)}")
print(f"  Max: {np.max(y)}")
print(f"  Mean: {np.mean(y)}")

# Check for non-zero content
zero_rows_X = np.all(X == 0, axis=1)
zero_rows_y = np.all(y == 0, axis=1)
print(f"Rows with all-zero X: {np.sum(zero_rows_X)} / {X.shape[0]}")
print(f"Rows with all-zero y: {np.sum(zero_rows_y)} / {X.shape[0]}")

# Print a non-zero sample
if np.any(~zero_rows_X):
    idx = np.where(~zero_rows_X)[0][0]
    print(f"Non-zero X sample (idx={idx}):")
    # 1-input combos are first 4*14*4 = 224 features.
    # Check 2-input combos starting at 224.
    print("2-input combo phases (indices 224-250):")
    print(X[idx, 224:250])
    
    # Check if there are ANY non-zero phases in the entire row
    phases_only = X[idx, :1470]
    print(f"Non-zero phases count: {np.count_nonzero(phases_only)}")
    print(f"Max phase value: {np.max(phases_only)}")
    
    print(f"Target y (idx={idx}):")
    print(y[idx])
else:
    print("ALL X ROWS ARE ZERO!")
    
if np.sum(zero_rows_y) == X.shape[0]:
    print("ALL Y ROWS ARE ZERO!")

# Check block structure
n_features = X.shape[1]
half = n_features // 2
print(f"Features: {n_features}, Half: {half}")

X_first = X[:, :half]
X_second = X[:, half:]

print("First half stats:")
print(f"  Max: {X_first.max()}")
print(f"  Mean: {X_first.mean()}")

print("Second half stats:")
print(f"  Max: {X_second.max()}")
print(f"  Mean: {X_second.mean()}")

# Check if first half looks like phases [0, 2pi]
is_phase_like = (X_first.max() <= 100) # strict check, usually < 7
print(f"First half phase-like (Max <= 100): {is_phase_like}")

# Check first few elements
print(f"X[0, :10]: {X[0, :10]}")

