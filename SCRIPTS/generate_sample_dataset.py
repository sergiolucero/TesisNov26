import numpy as np

# Generate sample 1D ADE dataset
# Features: x, Targets: u(x,t) solution samples

# Define grid
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 50)
X, T = np.meshgrid(x, t)

# Example analytical solution data: u = sin(pi*x)*exp(-t)
u = np.sin(np.pi*X)*np.exp(-T)

# Save as npz
np.savez_compressed('out/dataset.npz', x=x, t=t, u=u)

print('Sample dataset generated in out/dataset.npz')
