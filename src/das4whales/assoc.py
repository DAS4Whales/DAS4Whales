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
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.signal as sp
import das4whales as dw
import cmocean.cm as cmo
import matplotlib.colors as mcolors
import pandas as pd
from joblib import Parallel, delayed

## Main association function --------------------------------------------

def process_iteration(
    # Peak data
    n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    # Grid data
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x,
    # Cable positions
    n_cable_pos, s_cable_pos, n_longi_offset, s_longi_offset,
    # Association lists
    association_lists,
    # Hyperbolas
    hyperbolas,
    # Rejected lists
    rejected_lists,
    # Parameters
    fs, dt_kde, bin_width, dt_sel, w_eval, rms_threshold, c0, dx, dt_tol,
    # Iteration info
    iteration):
    """
    Process a single iteration of the peak association algorithm.
    
    Returns: Updated peak data, grid data, and association lists
    """
    # Unpack the association lists, hyperbolas, and rejected lists
    (nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair,
    nhf_assoc_list, shf_assoc_list, nlf_assoc_list, slf_assoc_list) = association_lists
    n_rejected_list, s_rejected_list, n_rejected_hyperbolas, s_rejected_hyperbolas = rejected_lists 
    n_used_hyperbolas, s_used_hyperbolas = hyperbolas

    # PART 1: PREPARE DATA AND COMPUTE KDEs
    # =====================================
    
    # Precompute the time indices from peaks for both frequency bands and cables
    n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs
    n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs
    s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs
    s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs

    # Calculate delayed picks for all grid points
    n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
    n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
    s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
    s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

    # Find the global min and max for KDE time range
    all_delayed_picks = [n_delayed_picks_hf, n_delayed_picks_lf, s_delayed_picks_hf, s_delayed_picks_lf]
    global_min = min(np.min(arr) for arr in all_delayed_picks)
    global_max = max(np.max(arr) for arr in all_delayed_picks)
    
    # Create time bins for KDE
    Nkde = np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
    t_kde = np.linspace(global_min, global_max, Nkde)

    # Compute KDEs in parallel for each type
    # North high frequency
    n_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRhf) 
        for i in range(n_shape_x)
    ))

    # North low frequency
    n_kde_lf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRlf)
        for i in range(n_shape_x)
    ))

    # South high frequency
    s_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRhf)
        for i in range(s_shape_x)
    ))

    # South low frequency
    s_kde_lf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRlf)
        for i in range(s_shape_x)
    ))

    # Reduced the number of grid points to speed up the process 
    if iteration == 0:  
        sum_kde = n_kde_hf + n_kde_lf + s_kde_hf + s_kde_lf
        maxsum = np.max(sum_kde, axis=1)
        binary = np.ones_like(maxsum)
        threshold = np.percentile(maxsum, 40)  # keep top 55%
        grid_mask = maxsum >= threshold
        n_arr_tg = n_arr_tg[grid_mask]
        s_arr_tg = s_arr_tg[grid_mask]
        n_shape_x = n_arr_tg.shape[0]
        s_shape_x = s_arr_tg.shape[0]

    # PART 2: FIND MAXIMA AND COMPUTE THEORETICAL ARRIVALS
    # ===================================================
    
    # Combine KDEs for high and low frequencies
    hf_kde = n_kde_hf + s_kde_hf  # Combined HF KDE from north and south
    lf_kde = n_kde_lf + s_kde_lf  # Combined LF KDE from north and south

    # Find maxima for HF KDE
    hf_max_idx = np.argmax(hf_kde)
    hf_imax, hf_tmax = np.unravel_index(hf_max_idx, hf_kde.shape)
    max_time_hf = t_kde[hf_tmax]

    # Find maxima for LF KDE
    lf_max_idx = np.argmax(lf_kde)
    lf_imax, lf_tmax = np.unravel_index(lf_max_idx, lf_kde.shape)
    max_time_lf = t_kde[lf_tmax]

    # Compute theoretical arrival times (hyperbolas)
    nhf_hyperbola = max_time_hf + n_arr_tg[hf_imax, :]  # North HF theoretical arrivals
    shf_hyperbola = max_time_hf + s_arr_tg[hf_imax, :]  # South HF theoretical arrivals
    nlf_hyperbola = max_time_lf + n_arr_tg[lf_imax, :]  # North LF theoretical arrivals
    slf_hyperbola = max_time_lf + s_arr_tg[lf_imax, :]  # South LF theoretical arrivals

    # PART 3: SELECT PICKS AND COMPUTE RESIDUALS
    # =========================================
    
    # Select picks around each hyperbola within +/- dt_sel
    nhf_idx_dist, nhf_idx_time = dw.assoc.select_picks(n_up_peaks_hf, nhf_hyperbola, dt_sel, fs)
    shf_idx_dist, shf_idx_time = dw.assoc.select_picks(s_up_peaks_hf, shf_hyperbola, dt_sel, fs)
    nlf_idx_dist, nlf_idx_time = dw.assoc.select_picks(n_up_peaks_lf, nlf_hyperbola, dt_sel, fs)
    slf_idx_dist, slf_idx_time = dw.assoc.select_picks(s_up_peaks_lf, slf_hyperbola, dt_sel, fs)

    # Calculate time indices
    nhf_times = nhf_idx_time / fs
    shf_times = shf_idx_time / fs
    nlf_times = nlf_idx_time / fs
    slf_times = slf_idx_time / fs

    # Define evaluation windows
    nhf_window_mask = dw.assoc.get_window_mask(nhf_times, w_eval)
    shf_window_mask = dw.assoc.get_window_mask(shf_times, w_eval)
    nlf_window_mask = dw.assoc.get_window_mask(nlf_times, w_eval)
    slf_window_mask = dw.assoc.get_window_mask(slf_times, w_eval)

    # Compute locations and residuals
    nhf_n, nhf_residuals = dw.assoc.loc_picks(nhf_idx_dist, nhf_idx_time, n_cable_pos, c0, fs)
    shf_n, shf_residuals = dw.assoc.loc_picks(shf_idx_dist, shf_idx_time, s_cable_pos, c0, fs)
    nlf_n, nlf_residuals = dw.assoc.loc_picks(nlf_idx_dist, nlf_idx_time, n_cable_pos, c0, fs)
    slf_n, slf_residuals = dw.assoc.loc_picks(slf_idx_dist, slf_idx_time, s_cable_pos, c0, fs)

    # Calculate RMS residuals
    nhf_rms = np.sqrt(np.mean(nhf_residuals[nhf_window_mask] ** 2))
    shf_rms = np.sqrt(np.mean(shf_residuals[shf_window_mask] ** 2))
    nlf_rms = np.sqrt(np.mean(nlf_residuals[nlf_window_mask] ** 2))
    slf_rms = np.sqrt(np.mean(slf_residuals[slf_window_mask] ** 2))

    # PART 4: ASSOCIATION LOGIC
    # ========================

    # Check all cases
    hf_north_south_good = nhf_rms < rms_threshold and shf_rms < rms_threshold
    only_hf_north_good = nhf_rms < rms_threshold and shf_rms >= rms_threshold
    only_hf_south_good = nhf_rms >= rms_threshold and shf_rms < rms_threshold
    
    lf_north_south_good = nlf_rms < rms_threshold and slf_rms < rms_threshold
    only_lf_north_good = nlf_rms < rms_threshold and slf_rms >= rms_threshold
    only_lf_south_good = nlf_rms >= rms_threshold and slf_rms < rms_threshold

    # HF and LF overlap
    if abs(max_time_hf - max_time_lf) < 1.4:
        if hf_kde[hf_imax, hf_tmax] > lf_kde[lf_imax, lf_tmax]:
            # HF is better
            lf_north_south_good = False
            only_lf_north_good = False
            only_lf_south_good = False
        else:
            # LF is better
            hf_north_south_good = False
            only_hf_north_good = False
            only_hf_south_good = False

    processed = False

    # Best case: Both HF and LF are good for both north and south
    # print(f"nhf_rms: {nhf_rms}, shf_rms: {shf_rms}, nlf_rms: {nlf_rms}, slf_rms: {slf_rms}")
    if hf_north_south_good and lf_north_south_good:
        # Process HF first (assuming it has priority)
        # North cable processing for HF
        #TODO: reselect picks using the new hyperbola ?
        if max_time_hf >= 0: # Do not associate the edge cases
            # snr = dw.assoc.select_snr(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, nSNRhf)
            mask_resi_n_hf = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
            # mask_resi_n_hf = np.ones_like(nhf_residuals, dtype=bool)
            nhf_assoc_list_pair.append(np.asarray((nhf_idx_dist[mask_resi_n_hf], nhf_idx_time[mask_resi_n_hf])))
            n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
            n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
            
            # South cable processing for HF
            mask_resi_s_hf = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
            # mask_resi_s_hf = np.ones_like(shf_residuals, dtype=bool)
            shf_assoc_list_pair.append(np.asarray((shf_idx_dist[mask_resi_s_hf], shf_idx_time[mask_resi_s_hf])))
            s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
            s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
        else:
            mask_resi_n_hf = np.one_like(nhf_residuals, dtype=bool)
            mask_resi_s_hf = np.one_like(shf_residuals, dtype=bool)
        
        # Then process LF
        if max_time_lf >= 0: # Do not associate the edge cases
            # North cable processing for LF
            mask_resi_n_lf = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
            # mask_resi_n_lf = np.ones_like(nlf_residuals, dtype=bool)
            nlf_assoc_list_pair.append(np.asarray((nlf_idx_dist[mask_resi_n_lf], nlf_idx_time[mask_resi_n_lf])))
            n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
            n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)
            
            # South cable processing for LF
            mask_resi_s_lf = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
            # mask_resi_s_lf = np.ones_like(slf_residuals, dtype=bool)
            slf_assoc_list_pair.append(np.asarray((slf_idx_dist[mask_resi_s_lf], slf_idx_time[mask_resi_s_lf])))
            s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
            s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
        else:
            mask_resi_n_lf = np.ones_like(nlf_residuals, dtype=bool)
            mask_resi_s_lf = np.ones_like(slf_residuals, dtype=bool)
        
        # Remove all selected picks from both frequencies and both cables
        # Accurate indexes 
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi_n_hf, nSNRhf)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi_n_lf, nSNRlf)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi_s_hf, sSNRhf)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi_s_lf, sSNRlf)

        # Fuzzy indexes (For peaks that are associated to hf or lf but also have points in the other band)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi_n_lf, nSNRhf, dt_tol=dt_tol)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi_n_hf, nSNRlf, dt_tol=dt_tol)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi_s_lf, sSNRhf, dt_tol=dt_tol)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi_s_hf, sSNRlf, dt_tol=dt_tol)

        processed = True

    # First priority: Case 1 - HF North and South are good
    elif hf_north_south_good:
        # North cable processing
        if max_time_hf >= 0: # Do not associate the edge cases
            mask_resi_n = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
            # mask_resi_n = np.ones_like(nhf_residuals, dtype=bool)
            nhf_assoc_list_pair.append(np.asarray((nhf_idx_dist[mask_resi_n], nhf_idx_time[mask_resi_n])))
            n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
            n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
        
            # South cable processing
            mask_resi_s = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
            # mask_resi_s = np.ones_like(shf_residuals, dtype=bool)
            shf_assoc_list_pair.append(np.asarray((shf_idx_dist[mask_resi_s], shf_idx_time[mask_resi_s])))
            s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
            s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
        else:
            mask_resi_n = np.ones_like(nhf_residuals, dtype=bool)
            mask_resi_s = np.ones_like(shf_residuals, dtype=bool)
        
        # Remove selected picks from both frequency bands (north)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi_n, nSNRhf)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi_n, nSNRlf, dt_tol=dt_tol)

        # Remove selected picks from both frequency bands (south)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi_s, sSNRhf)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi_s, sSNRlf, dt_tol=dt_tol)
        
        processed = True

    # Second priority: Case 2 - LF North and South are good  
    elif lf_north_south_good:
        # North cable processing
        if max_time_lf >= 0: # Do not associate the edge cases
            mask_resi_n = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
            # mask_resi_n = np.ones_like(nlf_residuals, dtype=bool)
            nlf_assoc_list_pair.append(np.asarray((nlf_idx_dist[mask_resi_n], nlf_idx_time[mask_resi_n])))
            n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
            n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)

            # South cable processing
            mask_resi_s = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
            # mask_resi_s = np.ones_like(slf_residuals, dtype=bool)
            slf_assoc_list_pair.append(np.asarray((slf_idx_dist[mask_resi_s], slf_idx_time[mask_resi_s])))
            s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
            s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
        else:
            mask_resi_n = np.ones_like(nlf_residuals, dtype=bool)
            mask_resi_s = np.ones_like(slf_residuals, dtype=bool)

        # Remove selected picks from both frequency bands (north)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi_n, nSNRlf)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi_n, nSNRhf, dt_tol=dt_tol)

        # Remove selected picks from both frequency bands (south)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi_s, sSNRlf)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi_s, sSNRhf, dt_tol=dt_tol)
        
        processed = True
    
    # Lower priority cases - if neither combined case is good, try individual cables
    if not processed:
        # Case 3: Only HF North is good
        if only_hf_north_good:
            if max_time_hf >= 0:
                mask_resi = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
                # mask_resi = np.ones_like(nhf_residuals, dtype=bool)
                nhf_assoc_list.append(np.asarray((nhf_idx_dist[mask_resi], nhf_idx_time[mask_resi])))
                n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
                n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
            else:
                mask_resi = np.ones_like(nhf_residuals, dtype=bool)

            n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi, nSNRhf)
            n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi, nSNRlf, dt_tol=dt_tol)
            processed = True
            
        # Case 4: Only HF South is good
        elif only_hf_south_good:
            if max_time_hf >= 0:
                mask_resi = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
                # mask_resi = np.ones_like(shf_residuals, dtype=bool)
                shf_assoc_list.append(np.asarray((shf_idx_dist[mask_resi], shf_idx_time[mask_resi])))
                s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
                s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
            else:
                mask_resi = np.ones_like(shf_residuals, dtype=bool)
            
            s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi, sSNRhf)
            s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi, sSNRlf, dt_tol=dt_tol)
            processed = True
            
        # Case 5: Only LF North is good
        elif only_lf_north_good:
            if max_time_lf >= 0:
                mask_resi = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
                # mask_resi = np.ones_like(nlf_residuals, dtype=bool)
                nlf_assoc_list.append(np.asarray((nlf_idx_dist[mask_resi], nlf_idx_time[mask_resi])))
                n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
                n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)
            else:
                mask_resi = np.ones_like(nlf_residuals, dtype=bool)
            
            n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi, nSNRlf)
            n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi, nSNRhf, dt_tol=dt_tol)
            processed = True
            
        # Case 6: Only LF South is good
        elif only_lf_south_good:
            if max_time_lf >= 0:
                mask_resi = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
                # mask_resi = np.ones_like(slf_residuals, dtype=bool)
                slf_assoc_list.append(np.asarray((slf_idx_dist[mask_resi], slf_idx_time[mask_resi])))
                s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
                s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
            else:
                mask_resi = np.ones_like(slf_residuals, dtype=bool)
            
            s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi, sSNRlf)
            s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi, sSNRhf, dt_tol=dt_tol)
            processed = True
    
    # Case 7: No good residuals - reject the hyperbolas
    if not processed:
        # Add the rejected hyperbolas to rejection lists
        n_rejected_list.append(np.asarray((nhf_idx_dist, nhf_idx_time)))
        n_rejected_list.append(np.asarray((nlf_idx_dist, nlf_idx_time)))
        n_rejected_hyperbolas.append(n_arr_tg[hf_imax, :])
        n_rejected_hyperbolas.append(n_arr_tg[lf_imax, :])
        
        s_rejected_list.append(np.asarray((shf_idx_dist, shf_idx_time)))
        s_rejected_list.append(np.asarray((slf_idx_dist, slf_idx_time)))
        s_rejected_hyperbolas.append(s_arr_tg[hf_imax, :])
        s_rejected_hyperbolas.append(s_arr_tg[lf_imax, :])
        
        # Remove the hyperbolas from the grid arrays
        n_arr_tg = np.delete(n_arr_tg, hf_imax, axis=0)
        s_arr_tg = np.delete(s_arr_tg, hf_imax, axis=0)
        n_shape_x = n_arr_tg.shape[0]
        s_shape_x = s_arr_tg.shape[0]

    # Update the progress bar with the number of associated calls
    association_lists = [
        nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair,
        nhf_assoc_list, shf_assoc_list, nlf_assoc_list, slf_assoc_list
        ]
    
    rejected_lists = [
        n_rejected_list, s_rejected_list, n_rejected_hyperbolas, s_rejected_hyperbolas
    ]

    hyperbolas = [
        n_used_hyperbolas, s_used_hyperbolas
    ]

      # Return all the updated data
    return (
        # Updated peak data
        n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
        nSNRhf, nSNRlf, sSNRhf, sSNRlf,
        n_arr_tg, s_arr_tg, n_shape_x, s_shape_x, 
        association_lists, rejected_lists, hyperbolas)

## Helper functions for KDE and peak selection ----------------------------


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


def select_picks(peaks, hyperbola, dt_sel, fs):
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


def loc_picks(idx_dist, idx_time, cable_pos, c0, fs, Nbiter=20):
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
    n, residuals = dw.loc.solve_lq_weight(times, cable_pos[idx_dist], c0, Nbiter, fix_z=True, ninit=init, residuals=True)
    
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


def get_window_mask(times, w_eval):
    """
    Returns a boolean mask for values in `times` that fall within a window
    starting at the minimum time and extending for `w_eval` units.

    Parameters
    ----------
    times : np.ndarray
        Array of time values (can be empty).
    w_eval : float
        Window duration.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `times`.
    """
    if times.size == 0:
        return np.zeros_like(times, dtype=bool)
    t0 = np.min(times)
    return (times >= t0) & (times < t0 + w_eval)



def apply_spatial_windows(peaks, snr, win):
    """
    Apply the spatial windows to the peaks.

    Parameters
    ----------
    peaks : tuple of np.ndarray
        The peaks indexes for the North and South cables.
    win : list of float
        The spatial window to apply.

    Returns
    -------
    tuple of np.ndarray
        The peaks indexes after applying the spatial window.
    """
    
    npeakshf, npeakslf, speakshf, speakslf = peaks
    nSNRhf, nSNRlf, sSNRhf, sSNRlf = snr
    
    # Apply the spatial window to the North cable peaks
    mask_hf = (npeakshf[0, :] >= win[0]) & (npeakshf[0, :] <= win[1])
    mask_lf = (npeakslf[0, :] >= win[0]) & (npeakslf[0, :] <= win[1])

    # Filter columns (preserve 2D structure)
    npeakshf = npeakshf[:, mask_hf]
    nSNRhf = nSNRhf[mask_hf]
    npeakslf = npeakslf[:, mask_lf]
    nSNRlf = nSNRlf[mask_lf]

    # Apply the spatial window to the South cable peaks
    mask_hf = (speakshf[0, :] >= win[0]) & (speakshf[0, :] <= win[1])
    mask_lf = (speakslf[0, :] >= win[0]) & (speakslf[0, :] <= win[1])

    speakshf = speakshf[:, mask_hf]
    sSNRhf = sSNRhf[mask_hf]
    speakslf = speakslf[:, mask_lf]
    sSNRlf = sSNRlf[mask_lf]

    peaks = (npeakshf, npeakslf, speakshf, speakslf)
    snr = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
    return peaks, snr


def filter_peaks(residuals, idx_dist, idx_time, longi_offset, dx, gap_tresh = 5):
    idxmin_t = np.argmin(idx_time)
    distances = (longi_offset + idx_dist) * dx * 1e-3

    mask_dist = abs(distances - distances[idxmin_t]) < 40 # Distance mask, 40 km from the minimum
    gaps = np.zeros_like(distances)

    rms_total = np.sqrt(np.mean(residuals**2))
    mask_resi = abs(residuals) <  1.5 * rms_total
    # Find the gaps only for the valid (masked) distances
    valid_distances = distances[mask_resi]
    if valid_distances.size > 1:
        gaps_valid = np.abs(np.diff(valid_distances))
        # Assign the gaps to the correct positions
        idx_valid = np.flatnonzero(mask_resi)
        gaps[idx_valid[:-1]] = gaps_valid

    # Distance gaps evaluation
    # gaps = np.diff(distances)
    # Remove points after a large gap from the minimum 
    for l, gap in enumerate(gaps[idxmin_t:]):
        if gap > gap_tresh:
            mask_resi[idxmin_t + l + 1:] = False
            break

    # Remove points before a large gap from the minimum, in the reverse direction
    for l, gap in enumerate(gaps[idxmin_t-1::-1]):
        if gap > gap_tresh:
            mask_resi[:idxmin_t - l] = False
            break
    return mask_resi & mask_dist


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


def clean_pairs(
    primary: list[np.ndarray],
    counterpart:list[np.ndarray],
    counterpart_associated: list[np.ndarray],
) -> None:
    
    """
    """
    empty_idx = [i for i, arr in enumerate(primary) if arr.size <= 1000]

    for i in reversed(empty_idx):
        counterpart_associated.append(counterpart.pop(i))
        primary.pop(i)
    return


def clean_singles(associtations: list[np.ndarray]) -> None:
    """
    Cleans the single associations by removing empty arrays.
    
    Parameters
    ----------
    associtations : list of np.ndarray
        List of associations to clean.
    
    Returns
    -------
    None
    """
    empty_idx = [i for i, arr in enumerate(associtations) if arr.size <= 1000]

    for i in reversed(empty_idx):
        associtations.pop(i)


def save_assoc(
    filename,
    pair_assoc, pair_loc,
    associations, localizations,
    n_used_hyperbolas, n_rejected_hyperbolas,
    s_used_hyperbolas, s_rejected_hyperbolas,
    n_rejected_list, s_rejected_list,
    n_ds, s_ds,
    dt_kde, bin_width, dt_tol,
    n_shape_x, s_shape_x,
    dt_sel, w_eval, iterations
):
    nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair = pair_assoc
    nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc = pair_loc
    nhf_associated_list, nlf_associated_list, shf_associated_list, slf_associated_list = associations
    nhf_localizations, nlf_localizations, shf_localizations, slf_localizations = localizations

    results = {
        "assoc_pair": {
            "north": {
                "hf": nhf_assoc_list_pair,
                "lf": nlf_assoc_list_pair
            },
            "south": {
                "hf": shf_assoc_list_pair,
                "lf": slf_assoc_list_pair
            }
        },
        "pair_loc": {
            "north": {
                "hf": nhf_pair_loc,
                "lf": nlf_pair_loc
            },
            "south": {
                "hf": shf_pair_loc,
                "lf": slf_pair_loc
            }
        },
        "assoc": {
            "north": {
                "hf": nhf_associated_list,
                "lf": nlf_associated_list
            },
            "south": {
                "hf": shf_associated_list,
                "lf": slf_associated_list
            }
        },
        "localizations": {
            "north": {
                "hf": nhf_localizations,
                "lf": nlf_localizations
            },
            "south": {
                "hf": shf_localizations,
                "lf": slf_localizations
            }
        },
        "hyperbolas": {
            "north": {
                "used": n_used_hyperbolas,
                "rejected": n_rejected_hyperbolas
            },
            "south": {
                "used": s_used_hyperbolas,
                "rejected": s_rejected_hyperbolas
            }
        },
        "rejected": {
            "north": n_rejected_list,
            "south": s_rejected_list    
        },
        "metadata": {
            "north": dict(n_ds.attrs),
            "south": dict(s_ds.attrs),
            "assoc_meta": {
                "dt_kde" : dt_kde,
                "bin_width" : bin_width,
                "dt_tol" : dt_tol,
                "n_shape_x" : n_shape_x,
                "s_shape_x" : s_shape_x,
                "dt_sel" : dt_sel,
                "w_eval" : w_eval,
                "iterations" : iterations
            }
        }
    }

    with open(filename, "wb") as f:
        pickle.dump(results, f)


## Plotting functions ----- ------------------------------------------------ 


def plot_peaks(peaks, SNR, selected_channels_m, dx, fs):
    nhf_peaks, nlf_peaks, shf_peaks, slf_peaks = peaks
    nhf_SNR, nlf_SNR, shf_SNR, slf_SNR = SNR
    n_selected_channels_m, s_selected_channels_m = selected_channels_m

    # Determine common color scale
    vmin = min(np.min(nhf_SNR), np.min(nlf_SNR), np.min(shf_SNR), np.min(slf_SNR))
    vmax = max(np.max(nhf_SNR), np.max(nlf_SNR), np.max(shf_SNR), np.max(slf_SNR))
    cmap = cm.plasma  # Define colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Normalize color range

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False)

    # First subplot
    sc1 = axes[0, 0].scatter(nhf_peaks[1][:] / fs, (n_selected_channels_m[0] + nhf_peaks[0][:] * dx) * 1e-3, 
                            c=nhf_SNR, cmap=cmap, norm=norm, s=nhf_SNR)
    axes[0, 0].set_title('North Cable - HF')
    axes[0, 0].set_ylabel('Distance [km]')
    axes[0, 0].grid(linestyle='--', alpha=0.5)

    # Second subplot
    sc2 = axes[0, 1].scatter(nlf_peaks[1][:] / fs, (n_selected_channels_m[0] + nlf_peaks[0][:] * dx) * 1e-3, 
                            c=nlf_SNR, cmap=cmap, norm=norm, s=nlf_SNR)
    axes[0, 1].set_title('North Cable - LF')
    axes[0, 1].grid(linestyle='--', alpha=0.5)

    # Third subplot
    sc3 = axes[1, 0].scatter(shf_peaks[1][:] / fs, (s_selected_channels_m[0] + shf_peaks[0][:] * dx) * 1e-3, 
                            c=shf_SNR, cmap=cmap, norm=norm, s=shf_SNR)
    axes[1, 0].set_title('South Cable - HF')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Distance [km]')
    axes[1, 0].grid(linestyle='--', alpha=0.5)

    # Fourth subplot
    sc4 = axes[1, 1].scatter(slf_peaks[1][:] / fs, (s_selected_channels_m[0] + slf_peaks[0][:] * dx) * 1e-3, 
                            c=slf_SNR, cmap=cmap, norm=norm, s=slf_SNR)
    axes[1, 1].set_title('South Cable - LF')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].grid(linestyle='--', alpha=0.5)

    # Create a single colorbar for all subplots
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('SNR')

    return fig


def plot_reject_pick(peaks, longi_offset, dist, dx, associated_list, rejected_list, rejected_hyperbolas, fs):
    # Plot the selected picks alongside the original picks
    plt.figure(figsize=(20,8))
    plt.subplot(2, 2, 1)
    plt.scatter(peaks[1][:] / fs, (longi_offset + peaks[0][:]) * dx * 1e-3, label='HF', s=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    plt.subplot(2, 2, 2)
    for select in associated_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]') 
    # Plot the deleted hyperbolas
    plt.subplot(2, 2, 3)
    for hyp in rejected_hyperbolas:
        plt.plot(hyp, dist/1e3, label='Rejected hyperbola')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    # plot the rejected picks
    plt.subplot(2, 2, 4)
    for select in rejected_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]')
    plt.show()


def plot_associated_bicable(n_peaks, s_peaks, longi_offset, pair_assoc_list, pair_loc_list, associated_list, localizations,
                            n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs):
    
    nhf_assoc_pair, nlf_assoc_pair, shf_assoc_pair, slf_assoc_pair = pair_assoc_list
    nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list = associated_list
    nhf_loc_pair, nlf_loc_pair, shf_loc_pair, slf_loc_pair = pair_loc_list
    nhf_localizations, nlf_localizations, shf_localizations, slf_localizations = localizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False, constrained_layout=True)

    # Get color palettes
    hf_palette = plt.get_cmap('YlOrRd_r')
    lf_palette = plt.get_cmap('YlGnBu_r')

    # Assign color per HF/LF event
    nbhf = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list) + len(shf_assoc_list)
    nblf = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list) + len(slf_assoc_list)

    start, end = 0.0, 0.6  # Avoids part of the coolormap that is too light

    hf_colors = [hf_palette(start + (end - start) * i / max(nbhf - 1, 1)) for i in range(nbhf)]
    lf_colors = [lf_palette(start + (end - start) * i / max(nblf - 1, 1)) for i in range(nblf)]

    # First subplot — North raw picks and associated
    # -- Raw picks --
    axes[0, 0].scatter(n_peaks[1][:] / fs, (longi_offset + n_peaks[0][:]) * dx * 1e-3,
                       label='All peaks', s=0.5, alpha=0.2, color='tab:grey', rasterized=True)
    # -- Associated picks - pairs --
    for i, select in enumerate(nhf_assoc_pair):
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(nlf_assoc_pair):
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        
    # -- Associated picks - single --
    for i, select in enumerate(nhf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair)
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
    for i, select in enumerate(nlf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair)
        # print(i, idx_offset, len(nhf_assoc_pair), len(shf_assoc_pair), len(nhf_assoc_list)print(len(lf_colors), len(nlf_assoc_list)))
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
    axes[0, 0].set_title('North')       
    axes[0, 0].set_ylabel('Distance [km]')
    axes[0, 0].set_xlim(0, 70)

    # Second subplot — North with arrival curves
    # -- Associated picks - pairs --
    for i, select in enumerate(nhf_assoc_pair):
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nhf_loc_pair[i][-1], n_cable_pos, 
                                                  nhf_loc_pair[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
                                                  
    for i, select in enumerate(nlf_assoc_pair):
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nlf_loc_pair[i][-1], n_cable_pos,
                                                  nlf_loc_pair[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    # -- Associated picks - single --
    for i, select in enumerate(nhf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair)
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nhf_localizations[i][-1], n_cable_pos,
                                                  nhf_localizations[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(nlf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair)
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nlf_localizations[i][-1], n_cable_pos,
                                                  nlf_localizations[i][:3], c0),
                        n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
    # Remove the y-axis ticks labels
    axes[0, 1].set_yticklabels([])

    # Third subplot — South raw picks and associated
    # -- Raw picks --
    axes[1, 0].scatter(s_peaks[1][:] / fs, (longi_offset + s_peaks[0][:]) * dx * 1e-3,
                       label='All peaks', s=0.5, alpha=0.2, color='tab:grey', rasterized=True)
    # -- Associated picks - pairs --
    for i, select in enumerate(shf_assoc_pair):
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(slf_assoc_pair):
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        
    # -- Associated picks - single --
    for i, select in enumerate(shf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list)
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(slf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list)
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
    axes[1, 0].set_title('South')
    axes[1, 0].set_ylabel('Distance [km]')
    axes[1, 0].set_xlabel('Time [s]')

    # Fourth subplot — South with arrival curves
    # -- Associated picks - pairs --
    for i, select in enumerate(shf_assoc_pair):
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(shf_loc_pair[i][-1], s_cable_pos,
                                                  shf_loc_pair[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(slf_assoc_pair):
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(slf_loc_pair[i][-1], s_cable_pos,
                                                  slf_loc_pair[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    # -- Associated picks - single --
    for i, select in enumerate(shf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list)
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(shf_localizations[i][-1], s_cable_pos,
                                                  shf_localizations[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(slf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list)
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(slf_localizations[i][-1], s_cable_pos,
                                                  slf_localizations[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)

    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_yticklabels([])

    # Add a common legend
    hf_handle = plt.Line2D([], [], marker='>', color='w', label='HF calls',
                           markerfacecolor='tab:red', markersize=10)
    lf_handle = plt.Line2D([], [], marker='o', color='w', label='LF calls',
                           markerfacecolor='tab:blue', markersize=10)
    

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    # Add gradient legend to one of your subplots
    gradient_values = np.linspace(start, end, 100).reshape(1, -1)
    hf_cmap_custom = ListedColormap(hf_colors)
    lf_cmap_custom = ListedColormap(lf_colors)

    # Create a parent container for the legend with frame
    legend_container = inset_axes(axes[1, 1], width="25%", height="20%", loc='lower right',
                                bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=axes[1, 1].transAxes)
    legend_container.set_xlim(0, 1)
    legend_container.set_ylim(0, 1)
    legend_container.set_xticks([])
    legend_container.set_yticks([])

    # Add frame around the container
    legend_container.spines['top'].set_visible(True)
    legend_container.spines['right'].set_visible(True)
    legend_container.spines['bottom'].set_visible(True)
    legend_container.spines['left'].set_visible(True)
    legend_container.spines['top'].set_linewidth(1.5)
    legend_container.spines['right'].set_linewidth(1.5)
    legend_container.spines['bottom'].set_linewidth(1.5)
    legend_container.spines['left'].set_linewidth(1.5)
    legend_container.spines['top'].set_color('black')
    legend_container.spines['right'].set_color('black')
    legend_container.spines['bottom'].set_color('black')
    legend_container.spines['left'].set_color('black')

    # HF gradient bar (positioned in upper part of container)
    hf_gradient_ax = inset_axes(legend_container, width="80%", height="35%", loc='upper center',
                            bbox_to_anchor=(0, 0.15, 1, 0.8), bbox_transform=legend_container.transAxes)
    hf_gradient_ax.imshow(gradient_values, aspect='auto', cmap=hf_cmap_custom)
    hf_gradient_ax.set_xticks([])
    hf_gradient_ax.set_yticks([])
    hf_gradient_ax.set_title('HF calls ▷', fontsize=12, pad=4)

    # LF gradient bar (positioned in lower part of container)
    lf_gradient_ax = inset_axes(legend_container, width="80%", height="35%", loc='lower center',
                            bbox_to_anchor=(0, -0.05, 1, 0.8), bbox_transform=legend_container.transAxes)
    lf_gradient_ax.imshow(gradient_values, aspect='auto', cmap=lf_cmap_custom)
    lf_gradient_ax.set_xticks([])
    lf_gradient_ax.set_yticks([])
    lf_gradient_ax.set_title('LF calls ●', fontsize=12, pad=4)
    for ax in axes.flat:
        ax.grid(linestyle='--', alpha=0.6)
    return fig


def plot_kdesurf(df_north: pd.DataFrame, df_south: pd.DataFrame, bathy: np.ndarray, 
                 x: np.ndarray, y: np.ndarray, xg: np.ndarray, yg: np.ndarray, 
                 heatmap: np.ndarray) -> plt.Figure:
    """
    """
    # Plot the grid points on the map
    colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
    colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

    # Combine the color maps
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

    extent = [x[0], x[-1], y[0], y[-1]]

    # Set the light source
    ls = LightSource(azdeg=350, altdeg=45)

    fig = plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

    # Plot the cable location in 2D
    ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
    ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

    # Plot the used cable locations
    # ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green', label='Used cable locations')

    # Plot the grid points
    ax.scatter(xg, yg, c='k', s=1)

    # Plot the heatmaps over the grid points
    ax.tricontourf(xg, yg, heatmap, levels=20, cmap='hot', alpha=0.5)

    # Use a proxy artist for the color bar
    im = ax.tricontourf(xg, yg, heatmap, levels=20, cmap='hot', alpha=0.5)

    im_ratio = bathy.shape[1] / bathy.shape[0]
    plt.colorbar(im, ax=ax, label='Standard deviaton', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
    im.remove()
    # Set the labels
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    return fig    