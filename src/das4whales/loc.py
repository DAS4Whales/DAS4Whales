"""
loc.py - Localisation module for the das4whales package.

This module provides functions for localizing the source of a sound source recorded by a DAS array.

Author: Quentin Goestchel, Léa Bouffaut
Date: 2024-06-18/2025-03-05
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple, Union, Optional, Any, NamedTuple

import numpy as np
from tqdm import tqdm

from das4whales.spatial import calc_das_section_bearing, calc_source_position_lat_lon, calc_dist_lat_lon


class LocalizationResult(NamedTuple):
    """Container for localization results with uncertainty information."""
    position: np.ndarray  # [x, y, z, t0]
    residuals: np.ndarray
    rms: float
    weighted_rms: float
    covariance: np.ndarray
    uncertainties: np.ndarray
    weights: np.ndarray
    n_iterations: int


def calc_arrival_times(t0: Union[float, np.ndarray], cable_pos: np.ndarray, pos: Union[Tuple[np.ndarray, np.ndarray, float], Tuple[float, float, float]], c0: float) -> np.ndarray:
    """
    Calculate theoretical arrival times of a whale call at a grid of positions or a single point.

    Parameters
    ----------
    t0 : float or np.ndarray
        Initial time offset. Can be a scalar or an array of time offsets.

    cable_pos : np.ndarray
        Array of cable positions with shape (N, 3), where N is the number of channels.
        Each row represents [x, y, z] coordinates of a cable position.

    pos : tuple of np.ndarray or float
        If a grid is used, provide a tuple (xg, yg, zg) with xg and yg as 2D arrays.  
        For a single point, provide (x, y, z) as floats or 1D arrays.

    c0 : float
        Speed of sound in water (in meters per second).

    Returns
    -------
    th_arrtimes : np.ndarray
        Theoretical arrival times at each grid point or point source.  
        Shape is (M, L, N) for a grid, or (N,) for a single point.
    """
    # Extract cable positions
    x_cable, y_cable, z_cable = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2]
    
    # Check if pos is a grid, a flattened grid or a single point
    if isinstance(pos[0], np.ndarray) and pos[0].ndim == 2:
        # Grid case (np.meshgrid)
        xg, yg, zg = pos
        x_exp = xg[:, :, np.newaxis]  # Shape (M, L, 1)
        y_exp = yg[:, :, np.newaxis]
        z_exp = zg  # Scalar or array

        # Calculate distances for grid
        dist = np.sqrt((x_cable[np.newaxis, np.newaxis, :] - x_exp) ** 2 +
                       (y_cable[np.newaxis, np.newaxis, :] - y_exp) ** 2 +
                       (z_cable[np.newaxis, np.newaxis, :] - z_exp) ** 2)
        
    elif isinstance(pos[0], np.ndarray) and pos[0].ndim == 1: #flattened grid case
        # Flattened grid case
        xg, yg, zg = pos
        x_exp = xg[:, np.newaxis]
        y_exp = yg[:, np.newaxis]
        z_exp = zg

        # Calculate distances for flattened grid
        dist = np.sqrt((x_cable[np.newaxis, :] - x_exp) ** 2 +
                       (y_cable[np.newaxis, :] - y_exp) ** 2 +
                       (z_cable[np.newaxis, :] - z_exp) ** 2)
        
    else:
        # Single point case
        x, y, z = pos
        dist = np.sqrt((x_cable - x) ** 2 +
                       (y_cable - y) ** 2 +
                       (z_cable - z) ** 2)

    # Calculate arrival times
    th_arrtimes = t0 + dist / c0

    return th_arrtimes


def calc_theory_toa(das_position, whale_position, dist, c0=1490):
    """
    Calculate theoretical Time of Arrival (TOA) for a whale call with known position relative to the cable.

    Args:
        das_position (dict): A dictionary containing latitude, longitude, and depth information of the cable (DAS) positions.
        whale_position (dict): A dictionary containing at least whale apex, offset, side and depth
        dist (numpy.ndarray): An array containing distances along the cable.
        c0 (float, optional): The speed of sound in water in meters per second. Defaults to 1490 m/s.

    Returns:
        numpy.ndarray: An array containing the theoretical TOAs for each position along the cable.
    """

    # Find the index of whale_apex_m in dist
    ind_whale_apex = np.where(dist >= whale_position['apex'])[0][0]

    # Get the bearing of the DAS cable around whale position
    step = 3
    das_bearing = calc_das_section_bearing(
        das_position['lat'][ind_whale_apex - step],
        das_position['lon'][ind_whale_apex - step],
        das_position['lat'][ind_whale_apex + step],
        das_position['lon'][ind_whale_apex + step])

    # Get the whale position
    whale_position['lat'], whale_position['lon'] = calc_source_position_lat_lon(
        das_position['lat'][ind_whale_apex],
        das_position['lon'][ind_whale_apex],
        whale_position['offset'],
        das_bearing,
        whale_position['side'])

    # Create an updated whale position dictionary
    # print(f"whale position {whale_position}")

    # Calculate 3D distance for each element in DAS_position
    distances = calc_dist_lat_lon(whale_position, das_position)

    # Calculate depth differences
    depth_diff = np.array(das_position['depth']) - whale_position['depth']

    # Calculate total distance (3D)
    total_distance = np.sqrt(distances ** 2 + depth_diff ** 2)

    # Calculate theoretical TOAs
    toa = (total_distance - np.min(total_distance)) / c0

    return toa


def calc_distance_matrix(cable_pos, whale_pos):
    """
    Compute the distance matrix between the cable and the whale
    """
    return np.sqrt(np.sum((cable_pos - whale_pos) ** 2, axis=1))


def calc_radii_matrix(cable_pos, whale_pos):
    """
    Compute the radii matrix between the cable and the whale
    """
    return np.sqrt(np.sum((cable_pos[:,:2] - whale_pos[:2]) ** 2, axis=1))


def calc_theta_vector(cable_pos, whale_pos):
    """
    Compute the elevation angle between the cable and the whale for each cable position
    """
    rj = calc_radii_matrix(cable_pos, whale_pos)
    return np.arctan2(abs(whale_pos[2]-cable_pos[:, 2]), rj)


def calc_phi_vector(cable_pos, whale_pos):
    """
    Compute the azimuthal angle between the cable and the whale for each cable position
    """
    return np.arctan2(whale_pos[1]-cable_pos[:,1], whale_pos[0]-cable_pos[:,0])


def solve_lq(Ti, cable_pos, c0, Nbiter=10,  SNR=None, fix_z=False, ninit=None, residuals=False, verbose=False):
    """
    Solve the least squares problem to localize the whale with optional SNR weighting
    
    Parameters
    ----------
    Ti : np.ndarray
        Array of arrival times at each cable position [channel x 1]
    cable_pos : np.ndarray
        Array of cable positions [channel x 3]
    c0 : float
        Speed of sound in water considered constant
    SNR : np.ndarray, optional
        Array of signal-to-noise ratios in dB for each measurement [channel x 1]
    Nbiter : int, optional (default=10)
        Number of iterations for the least squares algorithm
    fix_z : bool, optional (default=False)
        Whether to fix the z-coordinate
    ninit : np.ndarray, optional
        Initial guess for n
    residuals : bool, optional (default=False)
        Whether to return residuals
    verbose : bool, optional (default=False)
        Whether to print iteration details
        
    Returns
    -------
    n : np.ndarray
        Estimated whale position and time of emission vector [x, y, z, t0]
    res : np.ndarray, optional
        Residuals if residuals=True
    """
    # Make a first guess of the whale position
    n = np.array([40000, 1000, -30, np.min(Ti)])
    if ninit is not None:
        n = ninit
        
    # Regularization parameter
    lambda_reg = 1e-5
    
    # Create weight matrix from SNR values if provided
    if SNR is not None:
        # Convert SNR from dB to linear scale
        linear_snr = 10**(SNR/10)
        
        # Use SNR as weights - higher SNR = more weight
        W = np.diag(linear_snr)
        
        # Optional: normalize weights to sum to number of measurements
        # This keeps the overall influence of the regularization term similar
        W = W * (len(SNR) / np.sum(linear_snr))
    else:
        # Use uniform weights if no SNR provided
        W = np.eye(len(Ti))
    
    for j in range(Nbiter):
        thj = calc_theta_vector(cable_pos, n)
        phij = calc_phi_vector(cable_pos, n)
        dt = Ti - calc_arrival_times(n[-1], cable_pos, n[:3], c0)
        
        # Fixed z case
        if fix_z:
            # Save z value to reappend it after the least squares computation
            dz = n[2]
            n_fz = np.delete(n, 2)  # Remove z from the vector n
            del n  # Delete n to reassign it with the new value
            n = n_fz  # Reassign n without z
            
            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.ones_like(thj)]).T
        # Free z case
        else:
            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.sin(thj) / c0, np.ones_like(thj)]).T
        
        # Adding regularization to avoid singular matrix error
        lambda_identity = lambda_reg * np.eye(G.shape[1])
        
        # Weighted least squares solution: (G^T W G + λI)^(-1) G^T W dt
        dn = np.linalg.inv(G.T @ W @ G + lambda_identity) @ G.T @ W @ dt
        
        # Damping factor
        if j < 4:
            n += 0.7 * dn
        else:
            n += dn
            
        if fix_z:
            # reappend z to n in index 2 (before index 3)
            n = np.insert(n, 2, dz)
            
        if verbose:
            print(f'Iteration {j+1}: x = {n[0]:.4f} m, y = {n[1]:.4f}, z = {n[2]:.4f}, ti = {n[3]:.4f}')
    
    # Compute final residuals
    res = Ti - calc_arrival_times(n[-1], cable_pos, n[:3], c0)
    
    if residuals:
        return n, res
    else:
        return n


def calc_dist_weighting(dist, discut, disw1, disw2):
    # Initialize weights
    weights = np.zeros_like(dist)

    # Find second minimum distance
    if len(dist) > 1:
        dmin2 = np.partition(dist, 1)[1]
        
        if dmin2 > discut:  # Event outside the network
            # Set weights to 1 for distances less than dmin2 * disw1
            weights[dist < dmin2 * disw1] = 1
            
            # Apply cosine taper for distances between dmin2 * disw1 and dmin2 * disw2
            taper_indices = (dist >= dmin2 * disw1) & (dist < dmin2 * disw2)
            if np.any(taper_indices):
                weights[taper_indices] = 0.5 * (1 + np.cos(np.pi * (dist[taper_indices] - dmin2 * disw1) / 
                                                        (dmin2 * (disw2 - disw1))))
        else:  # Event inside the network
            # Set weights to 1 for distances less than discut * disw1
            weights[dist < discut * disw1] = 1
            
            # Apply cosine taper for distances between discut * disw1 and discut * disw2
            taper_indices = (dist >= discut * disw1) & (dist < discut * disw2)
            if np.any(taper_indices):
                weights[taper_indices] = 0.5 * (1 + np.cos(np.pi * (dist[taper_indices] - discut * disw1) / 
                                                        (discut * (disw2 - disw1))))

    # Create weight matrix: 
    return weights


def calc_res_weighting(res, rmscut, rmsw1, rmsw2):
    weights = np.zeros_like(res)
    rms = np.sqrt(np.mean(res**2))
    abs_res = np.abs(res)
    
    if rms > rmscut:  # poor solution
        # Set weights to 1 for residuals less than rmscut * rmsw1
        weights[abs_res < rms * rmsw1] = 1
        
        # Apply cosine taper for residuals between rmscut * rmsw1 and rmscut * rmsw2
        taper_indices = (abs_res >= rms * rmsw1) & (abs_res < rms * rmsw2)
        if np.any(taper_indices):
            weights[taper_indices] = 0.5 * (1 + np.cos(np.pi * (abs_res[taper_indices] - rms * rmsw1) / 
                                                    (rms * (rmsw2 - rmsw1))))
    else:  # good solution
        # Set weights to 1 for residuals less than rmscut * rmsw1
        weights[abs_res < rmscut * rmsw1] = 1
        
        # Apply cosine taper for residuals between rmscut * rmsw1 and rmscut * rmsw2
        taper_indices = (abs_res >= rmscut * rmsw1) & (abs_res < rmscut * rmsw2)
        if np.any(taper_indices):
            weights[taper_indices] = 0.5 * (1 + np.cos(np.pi * (abs_res[taper_indices] - rmscut * rmsw1) / 
                                                    (rmscut * (rmsw2 - rmsw1))))
    return weights


def solve_lq_weight(Ti, cable_pos, c0, Nbiter=10, fix_z=False, ninit=None, 
                   return_uncertainty=True, residuals=False, verbose=False):
    """
    Solve the least squares problem to localize the whale with distance weighting
    
    Parameters
    ----------
    Ti : np.ndarray
        Array of arrival times at each cable position [channel x 1]
    cable_pos : np.ndarray
        Array of cable positions [channel x 3]
    c0 : float
        Speed of sound in water considered constant
    Nbiter : int, optional (default=10)
        Number of iterations for the least squares algorithm
    fix_z : bool, optional (default=False)
        Whether to fix the z-coordinate
    ninit : np.ndarray, optional
        Initial guess for n
    return_uncertainty : bool, optional (default=True)
        Whether to compute and return uncertainty information
    residuals : bool, optional (default=False)
        Whether to return residuals (deprecated, use return_uncertainty)
    verbose : bool, optional (default=False)
        Whether to print iteration details
        
    Returns
    -------
    result : LocalizationResult or np.ndarray
        If return_uncertainty=True: LocalizationResult with comprehensive information
        If return_uncertainty=False: just the position array [x, y, z, t0]
    res : np.ndarray, optional
        Residuals if residuals=True (for backward compatibility)
    """
    # Make a first guess of the whale position
    n = np.array([40000, 1000, -30, np.min(Ti)])
    if ninit is not None:
        n = ninit
        
    # Regularization parameter
    lambda_reg = 1e-5
    # Distance weighting parameters
    discut = 10000 # 10 km
    disw1 = 1 # Cosine taper starts 
    disw2 = 3 # Cosine taper ends

    # Residual weighting parameters
    rmscut = 0.2 # 0.1 s
    rmsw1 = 1
    rmsw2 = 3
    
    # Store final weights for uncertainty analysis
    final_weights = None
    
    for j in range(Nbiter):
        thj = calc_theta_vector(cable_pos, n)
        phij = calc_phi_vector(cable_pos, n)
        dt = Ti - calc_arrival_times(n[-1], cable_pos, n[:3], c0)

        # Start the hypoinverse weighting only after 4 iterations
        if j < 4:
            W = np.eye(len(Ti))
        else:
            dist = calc_distance_matrix(cable_pos, n[:3])
            w = calc_dist_weighting(dist, discut, disw1, disw2) * calc_res_weighting(dt, rmscut, rmsw1, rmsw2)
            W = np.diag(w)
            final_weights = w  # Store for uncertainty analysis
        
        # Fixed z case
        if fix_z:
            # Save z value to reappend it after the least squares computation
            dz = n[2]
            n_fz = np.delete(n, 2)  # Remove z from the vector n
            del n  # Delete n to reassign it with the new value
            n = n_fz  # Reassign n without z
            
            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.ones_like(thj)]).T
        # Free z case
        else:
            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.sin(thj) / c0, np.ones_like(thj)]).T
        
        # Adding regularization to avoid singular matrix error
        lambda_identity = lambda_reg * np.eye(G.shape[1])
        
        # Weighted least squares solution: (G^T W G + λI)^(-1) G^T W dt
        dn = np.linalg.inv(G.T @ W @ G + lambda_identity) @ G.T @ W @ dt        

        # Damping factor
        if j < 4:
            n += 0.7 * dn # Value from USGS trial and error
        else:
            n += dn
            
        if fix_z:
            # reappend z to n in index 2 (before index 3)
            n = np.insert(n, 2, dz)
            
        if verbose:
            current_rms = np.sqrt(np.mean(dt**2))
            print(f'Iteration {j+1}: x = {n[0]:.4f} m, y = {n[1]:.4f}, z = {n[2]:.4f}, ti = {n[3]:.4f}, RMS = {current_rms:.6f}')
    
    # Compute final residuals
    res = Ti - calc_arrival_times(n[-1], cable_pos, n[:3], c0)
    
    if return_uncertainty:
        # Calculate RMS values
        rms_unweighted = calc_rms(res)
        
        # Calculate weighted RMS if weights are available
        if final_weights is not None:
            weighted_residuals = res * final_weights
            rms_weighted = np.sqrt(np.sum(final_weights * res**2) / np.sum(final_weights))
        else:
            rms_weighted = rms_unweighted
            final_weights = np.ones_like(res)
        
        # Calculate variance and covariance matrix
        predicted_times = calc_arrival_times(n[-1], cable_pos, n[:3], c0)
        
        # Use weighted variance if weights are available
        if final_weights is not None and not np.allclose(final_weights, final_weights[0]):
            # We have actual weights (not all equal)
            variance = cal_weighted_variance_residuals(Ti, predicted_times, final_weights, fix_z)
        else:
            # No weights or all weights equal - use unweighted variance
            variance = cal_variance_residuals(Ti, predicted_times, fix_z)
            final_weights = None  # Don't pass equal weights to covariance calculation
        
        covariance = calc_covariance_matrix(cable_pos, n, c0, variance, fix_z, final_weights)
        uncertainties = calc_uncertainty_position(cable_pos, n, c0, variance, fix_z, final_weights)
        
        result = LocalizationResult(
            position=n,
            residuals=res,
            rms=rms_unweighted,
            weighted_rms=rms_weighted,
            covariance=covariance,
            uncertainties=uncertainties,
            weights=final_weights,
            n_iterations=Nbiter
        )
        
        if residuals:  # Backward compatibility
            return result, res
        else:
            return result
    else:
        if residuals:
            return n, res
        else:
            return n
    

def cal_variance_residuals(arrtimes, predic_arrtimes, fix_z=False):
    """Compute the variance of the residuals of the arrival times

    Parameters
    ----------
    arrtimes : np.ndarray
        array of measured arrival times
    predic_arrtimes : np.ndarray
        array of predicted arrival times
    fix_z : bool, optional
        True if the z coordinate is fixed, by default False

    Returns
    -------
    var : float
        Variance of the residuals
    """
    residuals = arrtimes - predic_arrtimes
    if fix_z:
        var = 1 / (len(residuals) - 3) * np.sum(residuals**2)
    else:   
        var = 1 / (len(residuals) - 4) * np.sum(residuals**2)
    return var


def cal_weighted_variance_residuals(arrtimes, predic_arrtimes, weights, fix_z=False):
    """Compute the weighted variance of the residuals of the arrival times

    Parameters
    ----------
    arrtimes : np.ndarray
        array of measured arrival times
    predic_arrtimes : np.ndarray
        array of predicted arrival times
    weights : np.ndarray
        weights for each residual
    fix_z : bool, optional
        True if the z coordinate is fixed, by default False

    Returns
    -------
    var : float
        Weighted variance of the residuals
    """
    residuals = arrtimes - predic_arrtimes
    # Weighted variance formula: Σ(w_i * r_i²) / Σ(w_i)
    # But for uncertainty estimation, we need to account for degrees of freedom
    n_params = 3 if fix_z else 4
    effective_n = len(residuals) - n_params
    
    # Use effective sample size for weighted data
    # For weighted least squares, the variance estimate is:
    # σ² = Σ(w_i * r_i²) / (effective_n)
    var = np.sum(weights * residuals**2) / effective_n
    return var


def calc_covariance_matrix(cable_pos, whale_pos, c0, var, fix_z=False, weights=None):
    """Compute the covariance matrix of the estimated whale position

    Parameters
    ----------
    cable_pos : np.ndarray
        Array of cable positions [channel x 3]
    whale_pos : np.ndarray
        Estimated whale position [x, y, z, t0]
    c0 : float
        Speed of sound in water considered constant
    var : float
        Variance of the residuals (should be weighted variance if weights provided)
    fix_z : bool, optional
        Whether to fix the z coordinate (default: False)
    weights : np.ndarray, optional
        Weight vector for weighted least-squares (default: None for unweighted)

    Returns
    -------
    cov : np.ndarray
        Covariance matrix of the estimated whale position
    """
    thj = calc_theta_vector(cable_pos, whale_pos)
    phij = calc_phi_vector(cable_pos, whale_pos)

    if fix_z:
        G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.ones_like(thj)]).T
    else:
        G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.sin(thj) / c0, np.ones_like(thj)]).T

    # Apply weights if provided (weighted least-squares)
    if weights is not None:
        W = np.diag(weights)  # Convert to diagonal weight matrix
        GtWG = G.T @ W @ G
    else:
        GtWG = G.T @ G

    if np.linalg.cond(GtWG) > 1/sys.float_info.epsilon:
        print('Matrix is singular')
        lambda_reg = 1e-5
        lambda_identity = lambda_reg * np.eye(G.shape[1])
        cov = var * np.linalg.inv(GtWG + lambda_identity)
    else:
        cov = var * np.linalg.inv(GtWG)

    return cov


def calc_uncertainty_position(cable_pos, whale_pos, c0, var, fix_z=False, weights=None):
    """Compute the uncertainties on the estimated whale position

    Parameters
    ----------
    cable_pos : np.ndarray
        Array of cable positions [channel x 3]
    whale_pos : np.ndarray
        Estimated whale position [x, y, z, t0]
    c0 : float
        Speed of sound in water considered constant
    var : float
        Variance of the residuals (should be weighted variance if weights provided)
    fix_z : bool, optional
        Whether to fix the z coordinate (default: False)
    weights : np.ndarray, optional
        Weight vector for weighted least-squares (default: None for unweighted)

    Returns
    -------
    unc : np.ndarray
        Uncertainties on the estimated whale position
    """

    cov = calc_covariance_matrix(cable_pos, whale_pos, c0, var, fix_z, weights)
    unc = np.sqrt(np.diag(cov))

    return unc


def loc_from_picks(associated_list, cable_pos, c0, fs, return_uncertainty=True):
    """Localize whale calls from associated picks with uncertainty quantification.
    
    Parameters
    ----------
    associated_list : list
        List of associated picks
    cable_pos : np.ndarray
        Cable positions
    c0 : float
        Sound speed
    fs : float
        Sampling frequency
    return_uncertainty : bool, optional
        Whether to return uncertainty information (default=True)
        
    Returns
    -------
    list
        List of LocalizationResult objects if return_uncertainty=True,
        or list of position arrays if return_uncertainty=False
    """
    localizations = []

    for select in associated_list:
        idxmin_t = np.argmin(select[1][:])
        apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
        Ti = select[1][:] / fs
        Nbiter = 20

        # Initial guess (apex_loc, mean_y, -30m, min(Ti))
        n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]
        
        # Solve the least squares problem with uncertainty
        result = solve_lq_weight(Ti, cable_pos[select[0][:]], c0, Nbiter, 
                               fix_z=True, ninit=n_init, 
                               return_uncertainty=return_uncertainty)
        
        localizations.append(result)

    return localizations


def loc_picks_bicable(n_assoc, s_assoc, cable_pos, c0, fs, Nbiter=20, return_uncertainty=True):
    """
    Solve the least squares localization problem for bicable data using the picks' indices.
    
    Parameters
    ----------
    n_assoc : array-like
        The north cable association data [indices, times]
    s_assoc : array-like  
        The south cable association data [indices, times]
    cable_pos : tuple
        A tuple containing the positions of the north and south cables.
    c0 : float
        The speed of sound for localization.
    fs : float
        The sampling frequency.
    Nbiter : int, optional
        The number of iterations for the least squares solution, default is 20.
    return_uncertainty : bool, optional
        Whether to return uncertainty information (default=True)
    
    Returns
    -------
    LocalizationResult or tuple
        If return_uncertainty=True: LocalizationResult object
        If return_uncertainty=False: tuple (position, residuals) for backward compatibility
    """

    n_cable_pos, s_cable_pos = cable_pos
    bicable_pos = np.concatenate((n_cable_pos[n_assoc[0]], s_cable_pos[s_assoc[0]]))
    idx_time = np.concatenate((n_assoc[1], s_assoc[1]))
    idxmin_t = np.argmin(idx_time)  # Find the index of the minimum time

    times = idx_time / fs
    apex_loc = bicable_pos[idxmin_t, 0]  # Find the apex location from the minimum time index
    init = [apex_loc, np.mean(bicable_pos[:, 1]), -40, np.min(times)]  # Initial guess for the localization
    
    # Solve the least squares problem using the provided parameters
    if return_uncertainty:
        result = solve_lq_weight(times, bicable_pos, c0, Nbiter, fix_z=True, 
                               ninit=init, return_uncertainty=True)
        return result
    else:
        n, residuals = solve_lq_weight(times, bicable_pos, c0, Nbiter, fix_z=True, 
                                     ninit=init, return_uncertainty=False, residuals=True)
        return n, residuals


def loc_picks_bicable_list(n_assoc_list, s_assoc_list, cable_pos, c0, fs, Nbiter=20, return_uncertainty=True):
    """Localize multiple bicable associations with uncertainty quantification.
    
    Parameters
    ----------
    n_assoc_list : list
        List of north cable associations
    s_assoc_list : list
        List of south cable associations  
    cable_pos : tuple
        Cable positions for north and south
    c0 : float
        Sound speed
    fs : float
        Sampling frequency
    Nbiter : int, optional
        Number of iterations (default=20)
    return_uncertainty : bool, optional
        Whether to return uncertainty information (default=True)
        
    Returns
    -------
    list
        List of LocalizationResult objects if return_uncertainty=True,
        or list of position arrays if return_uncertainty=False
    """
    if len(n_assoc_list) != len(s_assoc_list):
        raise ValueError("The lengths of n_assoc_list and s_assoc_list must be equal.")

    localizations = []
    for i in range(len(n_assoc_list)):
        n_assoc = n_assoc_list[i]
        s_assoc = s_assoc_list[i]
        result = loc_picks_bicable(n_assoc, s_assoc, cable_pos, c0, fs, Nbiter, return_uncertainty)
        localizations.append(result)
    
    return localizations


def calc_rms(residuals, window_mask=None):
    """Calculate the root mean square of the residuals.

    Parameters
    ----------
    residuals : np.ndarray or None
        Array of residuals. If None or empty, returns np.inf.
    window_mask : np.ndarray, optional
        Boolean mask to apply to the residuals, by default None.
        If provided, only the residuals where the mask is True will be considered.

    Returns
    -------
    float
        Root mean square of the residuals, or np.inf if residuals are invalid or empty.
    """
    if residuals is None:
        return np.inf

    if window_mask is not None:
        residuals = residuals[window_mask]

    if residuals.size == 0:
        return np.inf

    return np.sqrt(np.mean(residuals ** 2))


def calc_weighted_rms(residuals, weights):
    """Calculate the weighted root mean square of the residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals.
    weights : np.ndarray
        Array of weights corresponding to each residual.

    Returns
    -------
    float
        Weighted root mean square of the residuals.
    """
    if residuals is None or weights is None:
        return np.inf
    
    if residuals.size == 0 or weights.size == 0:
        return np.inf
    
    if np.sum(weights) == 0:
        return np.inf

    return np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))


def calc_error_ellipse_params(covariance_matrix, confidence_level=0.95):
    """Calculate error ellipse parameters from covariance matrix.
    
    Parameters
    ----------
    covariance_matrix : np.ndarray
        Covariance matrix (at least 2x2 for x,y components)
    confidence_level : float, optional
        Confidence level (default 0.95 for 95% confidence)
        
    Returns
    -------
    dict
        Dictionary containing ellipse parameters:
        - semi_major_axis: length of semi-major axis
        - semi_minor_axis: length of semi-minor axis  
        - rotation_angle: rotation angle in radians
        - area: ellipse area
    """
    # Extract 2x2 covariance matrix for x,y coordinates
    cov_xy = covariance_matrix[:2, :2]
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_xy)
    
    # Sort by eigenvalue (largest first)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Chi-squared value for confidence level (use a fallback if scipy not available)
    try:
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence_level, df=2)
    except ImportError:
        # Fallback approximation for 95% confidence
        if confidence_level == 0.95:
            chi2_val = 5.991  # chi2(0.95, df=2)
        elif confidence_level == 0.68:
            chi2_val = 2.279  # chi2(0.68, df=2) 
        else:
            chi2_val = 5.991  # Default to 95%
    
    # Calculate ellipse parameters
    semi_major_axis = np.sqrt(chi2_val * eigenvals[0])
    semi_minor_axis = np.sqrt(chi2_val * eigenvals[1])
    
    # Rotation angle (angle of major axis with x-axis)
    rotation_angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Ellipse area
    area = np.pi * semi_major_axis * semi_minor_axis
    
    return {
        'semi_major_axis': semi_major_axis,
        'semi_minor_axis': semi_minor_axis,
        'rotation_angle': rotation_angle,
        'area': area,
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs
    }


def localization_results_to_dict(results, utc_start=None, sensor='unknown', call_type='unknown'):
    """Convert LocalizationResult objects to dictionary format for CSV export.
    
    Parameters
    ----------
    results : list of LocalizationResult
        List of localization results
    utc_start : datetime, optional
        Start time for converting relative times to absolute UTC
    sensor : str, optional
        Sensor identifier (default='unknown')
    call_type : str, optional
        Call type identifier (default='unknown')
        
    Returns
    -------
    list of dict
        List of dictionaries ready for DataFrame conversion
    """
    import datetime
    
    rows = []
    for i, result in enumerate(results):
        if isinstance(result, LocalizationResult):
            pos = result.position
            ellipse_params = calc_error_ellipse_params(result.covariance)
            
            row = {
                'id': i,
                'utc': utc_start + datetime.timedelta(seconds=float(pos[3])) if utc_start else None,
                'sensor': sensor,
                'call_type': call_type,
                'x': float(pos[0]),
                'y': float(pos[1]), 
                'z': float(pos[2]),
                't0': float(pos[3]),
                'rms': float(result.rms),
                'weighted_rms': float(result.weighted_rms),
                'unc_x': float(result.uncertainties[0]),
                'unc_y': float(result.uncertainties[1]),
                'unc_z': float(result.uncertainties[2]) if len(result.uncertainties) > 2 else np.nan,
                'unc_t': float(result.uncertainties[-1]),
                'ellipse_semi_major': float(ellipse_params['semi_major_axis']),
                'ellipse_semi_minor': float(ellipse_params['semi_minor_axis']),
                'ellipse_rotation': float(ellipse_params['rotation_angle']),
                'ellipse_area': float(ellipse_params['area']),
                'n_picks': len(result.residuals),
                'n_iterations': result.n_iterations
            }
        else:
            # Handle backward compatibility with simple arrays
            pos = result if hasattr(result, '__len__') else [result, 0, 0, 0]
            row = {
                'id': i,
                'utc': utc_start + datetime.timedelta(seconds=float(pos[3])) if utc_start and len(pos) > 3 else None,
                'sensor': sensor,
                'call_type': call_type,
                'x': float(pos[0]) if len(pos) > 0 else np.nan,
                'y': float(pos[1]) if len(pos) > 1 else np.nan,
                'z': float(pos[2]) if len(pos) > 2 else np.nan,
                't0': float(pos[3]) if len(pos) > 3 else np.nan,
                'rms': np.nan,
                'weighted_rms': np.nan,
                'unc_x': np.nan,
                'unc_y': np.nan,
                'unc_z': np.nan,
                'unc_t': np.nan,
                'ellipse_semi_major': np.nan,
                'ellipse_semi_minor': np.nan,
                'ellipse_rotation': np.nan,
                'ellipse_area': np.nan,
                'n_picks': np.nan,
                'n_iterations': np.nan
            }
        rows.append(row)
    
    return rows
    

    