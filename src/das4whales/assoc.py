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
        kde = gaussian_kde(delayed_picks, bw_method=bin_width/np.std(delayed_picks), weights=weights)
        density = kde(t_kde)
    else:
        kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
        log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        density = np.exp(log_dens) # Convert log-density to normal density
    return density


def fast_kde_rect(delayed_picks, t_kde, overlap=None, bin_width=None, weights=None):
    """
    Fast KDE approximation using histogram and optional rectangular smoothing.
    
    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    """
    # Histogram the picks
    hist_range = (t_kde[0], t_kde[-1])
    bins = len(t_kde)
    
    hist, _ = np.histogram(delayed_picks, bins=bins, range=hist_range, weights=weights)
    
    # Optional rectangular smoothing
    if overlap is None:
        overlap = np.diff(t_kde).mean()
    if bin_width is None:
        bin_width = 2 * overlap
    kernel_bins = int(np.round(bin_width / overlap))
    if kernel_bins % 2 == 0:
        kernel_bins += 1  # Ensure odd length
    kernel = np.ones(kernel_bins) / kernel_bins
    hist = sp.convolve(hist, kernel, mode="same")
    
    return hist / np.trapezoid(hist, t_kde)  # Normalize to match KDE style


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

    return up_peaks, arr_tg, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas\
    

def loc_from_picks(idx_dist, idx_time, cable_pos, c0, fs, Nbiter=20):
    """
    Solve the least squares localization problem for a single cable using the picks' indices.
    
    Parameters
    ----------
    idx_dist : array-like
        The indices for the cable positions.
    idx_time : array-like
        The times corresponding to the cable positions.
    cable_pos : array-like
        The positions of the cable.
    c0 : float
        The speed of sound or another relevant constant for localization.
    fs : float
        The sampling frequency.
    Nbiter : int, optional
        The number of iterations for the least squares solution, default is 20.
    
    Returns
    -------
    tuple
        A tuple containing the solution and the residuals of the least squares problem.
    """
    idxmin_t = np.argmin(idx_time)  # Find the index of the minimum time
    times = idx_time / fs
    apex_loc = cable_pos[:, 0][idx_dist[idxmin_t]]  # Find the apex location from the minimum time index
    init = [apex_loc, np.mean(cable_pos[:, 1]), -40, np.min(times)]  # Initial guess for the localization
    
    # Solve the least squares problem using the provided parameters
    n, residuals = dw.loc.solve_lq(times, cable_pos[idx_dist], c0, Nbiter, fix_z=True, ninit=init, residuals=True)
    
    return n, residuals


def compute_cumsum(residuals, idx_t, threshold=1500):
    """
    Compute a mask based on the cumulative sum of residuals.

    Parameters
    ----------
    residuals : array-like
        The residuals for the localization.
    idx_t : array-like
        The time indices for the residuals.
    threshold : float, optional
        The threshold value for the cumulative sum to generate the mask (default is 1500).

    Returns
    -------
    mask_resi : array-like
        A boolean mask where cumulative residuals are less than the threshold.
    """
    idx_min_t = np.argmin(idx_t)  # Find the index of the minimum time
    left_cs = np.cumsum(abs(residuals[idx_min_t::-1]))  # Cumulative sum for the left side
    right_cs = np.cumsum(abs(residuals[idx_min_t:]))   # Cumulative sum for the right side
    mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))  # Combine both sides
    mask_resi = mod_cs < threshold  # Create the mask based on the threshold
    
    return mask_resi


def select_snr(up_peaks, selected_peaks, snr):
    print(np.shape(up_peaks), np.shape(selected_peaks), np.shape(snr))
    # Start with a mask of all True
    mask = np.zeros(up_peaks.shape[1], dtype=bool)

    # Accumulate the mask for each selected pair (d, t)
    for d, t in zip(selected_peaks[0], selected_peaks[1]):
        mask |= (up_peaks[0, :] == d) & (up_peaks[1, :] == t)

    # Return the snr values for the selected (d, t) pairs
    return snr[mask]


def remove_peaks(up_peaks, idx_dist, idx_time, mask_resi, snr):
    mask = np.ones(up_peaks.shape[1], dtype=bool)
    for d, t in zip(idx_dist[mask_resi], idx_time[mask_resi]):
        mask &= ~((up_peaks[0, :] == d) & (up_peaks[1, :] == t))
    return up_peaks[:, mask], snr[mask]


def remove_peaks_tolerance(up_peaks, idx_dist, idx_time, mask_resi, snr, dist_tol=0, dt_tol=10):
    """
    Removes peaks from up_peaks that are close (in index space) to (dist, time) values.

    Parameters
    ----------
    up_peaks : np.ndarray
        2 x N array of peaks (row 0: dist_idx, row 1: time_idx).
    idx_dist : np.ndarray
        Array of distance indices of picks to compare.
    idx_time : np.ndarray
        Array of time indices of picks to compare.
    mask_resi : np.ndarray of bool
        Mask to select which (idx_dist, idx_time) pairs to use.
    snr : np.ndarray
        SNR values associated with each peak (same length as up_peaks.shape[1]).
    dist_tol : int
        Distance tolerance in index units.
    dt_tol : int
        Time tolerance in index units.

    Returns
    -------
    up_peaks_new : np.ndarray
        Filtered up_peaks with nearby picks removed.
    snr_new : np.ndarray
        Filtered SNR array.
    """
    mask = np.ones(up_peaks.shape[1], dtype=bool)

    for d, t in zip(idx_dist[mask_resi], idx_time[mask_resi]):
        dist_match = np.abs(up_peaks[0, :] - d) <= dist_tol
        time_match = np.abs(up_peaks[1, :] - t) <= dt_tol
        mask &= ~(dist_match & time_match)

    return up_peaks[:, mask], snr[mask]
