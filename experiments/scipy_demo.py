"""Small SciPy experiment for UAV-AERO ENGINE V0.1."""

import numpy as np
from scipy import optimize

def objective(x):
    """Simple objective function for testing optimization."""
    return (x - 3) ** 2 + 2

result = optimize.minimize(objective, x0=np.array([0.0]))
print("Scipy optimization result:")
print(f"Optimal x: {result.x[0]:.4f}")
print(f"Mimimum value: {result.fun:.4f}")
print(f"Success: {result.success}")