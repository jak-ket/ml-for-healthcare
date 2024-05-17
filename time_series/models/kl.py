import numpy as np
from scipy.stats import gaussian_kde

def kl_divergence(p, q):
    # Ensure the densities are positive
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return np.sum(p * np.log(p / q))

def estimate_density(data, points):
    kde = gaussian_kde(data.T)
    return kde(points.T)

def compute_kl_divergence(data1, data2):
    # Create grid points where we estimate the densities
    min_values = np.minimum(np.min(data1, axis=0), np.min(data2, axis=0))
    max_values = np.maximum(np.max(data1, axis=0), np.max(data2, axis=0))

    num_points = 100  # Number of grid points in each dimension
    grid_points = [np.linspace(min_val, max_val, num_points) for min_val, max_val in zip(min_values, max_values)]
    grid = np.meshgrid(*grid_points)
    grid = np.vstack([g.ravel() for g in grid]).T
    
    # Estimate densities
    p = estimate_density(data1, grid)
    q = estimate_density(data2, grid)
    
    # Compute KL divergence
    kl_div = kl_divergence(p, q)
    
    return kl_div