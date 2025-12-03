
import numpy as np
import astropy.units as u
from phise import Context
from src.analysis import data_representations
import matplotlib.pyplot as plt

# Disable plots
plt.show = lambda: None

def verify():
    print("Running instant_distribution verification...")
    
    # Setup context as in the notebook
    ctx = Context.get_VLTI()
    ctx.interferometer.chip.σ = np.zeros(14) * u.um
    ctx.target.companions[0].c = 0.1
    
    # Fix seed for reproducibility
    np.random.seed(42)
    
    # Run with smaller n for speed, but enough for checking logic
    n = 100
    
    print(f"Computing distribution with n={n}...")
    data, data_so = data_representations.instant_distribution(
        ctx=ctx, 
        n=n, 
        show=False, 
        compare=True
    )
    
    print("Computation done.")
    
    # Calculate simple stats to verify
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    
    print(f"Mean Data: {mean_data}")
    print(f"Std Data: {std_data}")
    
    return mean_data, std_data

if __name__ == "__main__":
    verify()
