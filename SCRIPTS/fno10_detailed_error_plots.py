import numpy as np
import matplotlib.pyplot as plt
import os

outdir = 'out/fno10'

# This script assumes you have loaded or saved these datasets from your FNO training pipeline
# Replace these dummy loads with actual data loading as needed

# Dummy load functions
# Load true outputs and preds from saved npz or generated files
# For example:
# data = np.load(os.path.join(outdir, 'test_results.npz'))
# true_u = data['true_u']
# pred_u = data['pred_u']

# Here, replace with actual loading logic for your test data

# This is placeholder sample data shape (2 test samples, nt=50, nx=100)
true_u = np.random.rand(2, 50, 100)
pred_u = true_u + 0.1*np.random.randn(2, 50, 100)  # Add noise as example

# Compute error magnitudes
errors = np.abs(pred_u - true_u)

# Plot error histograms for each test sample
plt.figure(figsize=(12, 6))
for i in range(errors.shape[0]):
    plt.subplot(1, errors.shape[0], i+1)
    plt.hist(errors[i].ravel(), bins=50, color='c', alpha=0.7)
    plt.title(f'Error histogram sample {i}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'error_histograms.png'))
plt.close()

# Visualize predictions vs truth for each test sample
for i in range(errors.shape[0]):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(true_u[i], aspect='auto')
    plt.title('Ground Truth U')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(pred_u[i], aspect='auto')
    plt.title('Predicted U')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(errors[i], aspect='auto')
    plt.title('Absolute Error')
    plt.colorbar()

    plt.suptitle(f'Test Sample {i}')
    plt.savefig(os.path.join(outdir, f'prediction_vs_truth_sample_{i}.png'))
    plt.close()

print(f'Detailed error diagnostics and visualization plots saved to {outdir}')
