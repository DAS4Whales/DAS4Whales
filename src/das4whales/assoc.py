"""
assoc.py - Association functions for DAS data processing

This module provides functions to associate picked times from fin whale calls, gatherd from DAS data.

Authors: Quentin Goestchel
Date: 2024
"""

import numpy as np
import das4whales as dw
from sklearn.neighbors import KernelDensity


def compute_kde(delayed_picks, t_kde, bin_width):
    """Computes the KDE of the delayed picks.

    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    bin_width : float
        Bin width for the KDE.

    Returns
    -------
    array-like
        KDE density values.  
    
    """

    # kde = gaussian_kde(delayed_picks, bw_method=bin_width / (np.max(t_kde) - np.min(t_kde)))
    kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
    kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
    log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
    return np.exp(log_dens) # Convert log-density to normal density


def compute_selected_picks(peaks, hyperbola, dt_sel, fs):
    """Selects picks that are closest to the hyperbola within a given time window."""
    selected_picks = ([], [])
    for i, idx in enumerate(peaks[1]):
        dist_idx = peaks[0][i]
        pick_time = idx / fs

        if hyperbola[dist_idx] - dt_sel < pick_time < hyperbola[dist_idx] + dt_sel:
            if dist_idx in selected_picks[0]:
                existing_idx = selected_picks[0].index(dist_idx)
                if abs(hyperbola[dist_idx] - pick_time) < abs(hyperbola[dist_idx] - selected_picks[1][existing_idx] / fs):
                    selected_picks[1][existing_idx] = idx  # Replace with closer pick
            else:
                selected_picks[0].append(dist_idx)
                selected_picks[1].append(idx)
    
    return np.array(selected_picks[0]), np.array(selected_picks[1])


def compute_curvature(w_times, w_distances):
    """Computes curvature using second derivatives."""
    ddx = np.diff(w_times)
    ddy = np.diff(w_distances)
    ddx2 = np.diff(ddx)
    ddy2 = np.diff(ddy)
    curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
    return np.mean(curvature)