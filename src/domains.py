"""Evaluation domains.

Try to keep objectives to the range 1.0, or at least with a maximum value of
1.0.
"""
import numpy as np
import torch

def arm(solutions: torch.Tensor):
    """Arm repertoire domain in PyTorch.

    The length of each arm link is 1.0.

    Args:
        solutions (torch.Tensor): A (batch_size, solution_dim) array where each
            row contains the joint angles for the arm.
    Returns:
        objectives (torch.Tensor): (batch_size,) array of objectives.
        features (torch.Tensor): (batch_size, 2) array of features.
    """
    objectives = 1.0 - torch.var(solutions, axis=1)
    cum_theta = torch.cumsum(solutions, axis=1)

    # Assumes link lengths are 1.0.
    x_pos = torch.cos(cum_theta)
    y_pos = torch.sin(cum_theta)

    features = torch.cat(
        (
            torch.sum(x_pos, axis=1, keepdims=True),
            torch.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objectives, features

def sphere(solutions: torch.Tensor):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solutions (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    dim = solutions.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = torch.sum(torch.square(solutions - sphere_shift), axis=1)
    objectives = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate measures.
    clipped = solutions.clone()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    features = torch.cat(
        (
            torch.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            torch.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objectives, features

# Config for the domains. All values should be JSON-compatible.
DOMAIN_CONFIGS = {
    "arm_10d": {
        "name": "arm_10d",
        "obj_meas_func": arm,
        "solution_dim": 10,
        "solution_bounds": [(-np.pi, np.pi)] * 10,
        "initial_sol": np.zeros(10).tolist(),
        "feature_low": [-10, -10],
        "feature_high": [10, 10],
    },
    "arm_100d": {
        "name": "arm_100d",
        "obj_meas_func": arm,
        "solution_dim": 100,
        "solution_bounds": [(-np.pi, np.pi)] * 100,
        "initial_sol": np.zeros(100).tolist(),
        "feature_low": [-100, -100],
        "feature_high": [100, 100],
    },
    "sphere_100d": {
        "name": "sphere_100d",
        "obj_meas_func": sphere,
        "solution_dim": 100,
        "solution_bounds": None,
        "initial_sol": np.zeros(100).tolist(),
        "feature_low": [-100 / 2 * 5.12, -100 / 2 * 5.12],
        "feature_high": [100 / 2 * 5.12, 100 / 2 * 5.12],
    }
}