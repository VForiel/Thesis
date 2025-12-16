
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

file_path = Path('generated/dataset/phase_maps/dataset_ns10000_nst20_gamma9.npz')
print(f"Loading {file_path}...")
data = np.load(file_path)
X = data['X']
y = data['y']

print(f"X shape: {X.shape}")

# Calculate variance of X across samples (axis 0)
x_var = np.var(X, axis=0)
print(f"X variance stats:")
print(f"  Min: {np.min(x_var)}")
print(f"  Max: {np.max(x_var)}")
print(f"  Mean: {np.mean(x_var)}")
print(f"  Non-zero variance features: {np.count_nonzero(x_var > 1e-10)} / {X.shape[1]}")

# Check if rows are identical
diff = np.abs(X[0] - X[1])
print(f"Diff between sample 0 and 1: Max={np.max(diff)}, Mean={np.mean(diff)}")

if np.max(x_var) < 1e-10:
    print("CRITICAL: Dataset has ZERO variance across samples! Inputs are constant.")
else:
    print("Dataset has variance.")
