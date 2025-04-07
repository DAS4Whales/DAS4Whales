"""
assoc.py - Association functions for DAS data processing

This module provides functions to associate picked times from fin whale calls, gatherd from DAS data.

Authors: Quentin Goestchel
Date: 2024
"""

import numpy as np
import das4whales as dw
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


def compute_kde(delayed_picks, t_kde, bin_width, weights=None):
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
    if weights is not None:
        # Use weighted KDE, Scipy's gaussian_kde is faster that sklearn's KernelDensity for weighted KDE
        kde = gaussian_kde(delayed_picks, bw_method=bin_width / (np.max(t_kde) - np.min(t_kde)), weights=weights)
        density = kde(t_kde)
    else:
        kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
        log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        density = np.exp(log_dens) # Convert log-density to normal density
    return density


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


def associate_picks(kde, t_grid, longi_offset, up_peaks, arr_tg, dx, c0, w_eval, dt_sel, fs, cable_pos, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas):
    """Associates picks with hyperbolas and updates the picks list."""
    # Find the maximum of the KDE
    max_kde_idx = np.argmax(kde)
    imax, tmax = np.unravel_index(max_kde_idx, kde.shape)
    max_time = t_grid[imax, tmax].item()
    # Select the picks that are within the 1.4 s window of the hyperbola
    hyperbola = max_time + arr_tg[imax, :] # Theoretical arrival times for the selected hyperbola
    idx_dist, idx_time = compute_selected_picks(up_peaks, hyperbola, dt_sel, fs) # Select the picks around the hyperbola within +/- dt_sel

    times = idx_time / fs
    distances = (longi_offset + idx_dist) * dx * 1e-3

    window_mask = (times > np.min(times)) & (times < np.min(times) + w_eval)
    w_times = times[window_mask]
    w_distances = distances[window_mask]

    # Calulate least squares fit
    idxmin_t = np.argmin(idx_time)
    apex_loc = cable_pos[:, 0][idx_dist[idxmin_t]]
    Ti = idx_time / fs
    Nbiter = 20
    # Initial guess (apex_loc, mean_y, -30m, min(Ti))
    n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

    # Solve the least squares problem
    n, residuals = dw.loc.solve_lq(Ti, cable_pos[idx_dist], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
    # rms residual
    rms = np.sqrt(np.mean(residuals[window_mask]**2))
    
    if rms < .5:
        # Compute the residual cumsum from the minimum time, in positive and negative directions
        #TODO: change variable names
        left_cs = np.cumsum(abs(residuals[idxmin_t::-1])) # negative direction
        right_cs = np.cumsum(abs(residuals[idxmin_t:])) # positive direction
        mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))

        mask_resi = mod_cs < 1500 # Mask the residuals that are below the threshold, key parameter

        associated_list.append(np.asarray((idx_dist[mask_resi], idx_time[mask_resi])))
        used_hyperbolas.append(arr_tg[imax, :])
        arr_tg[imax, :] = dw.loc.calc_arrival_times(0, cable_pos, n[:3], c0)

        # Remove selected picks from updated picks
        # Create a boolean mask that starts by marking every column as True (to keep)
        mask = np.ones(up_peaks.shape[1], dtype=bool)
        for d, t in zip(idx_dist[mask_resi], idx_time[mask_resi]):   # For each pair to remove, update the mask
            mask &= ~((up_peaks[0, :] == d) & (up_peaks[1, :] == t))
        # Apply the mask only once to filter the columns
        up_peaks = up_peaks[:, mask]

    # if compute_curvature(w_times, w_distances) < 1000:
    #     associated_list.append(np.asarray((sidx_dist, sidx_time)))
    #     used_hyperbolas.append(arr_tg[imax, :])

    else:
        # Add the rejected hyperbola to the list
        rejected_list.append(np.asarray((idx_dist, idx_time)))
        rejected_hyperbolas.append(arr_tg[imax, :])
        # Remove the hyperbola from the list
        arr_tg = np.delete(arr_tg, imax, axis=0)

    return up_peaks, arr_tg, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas