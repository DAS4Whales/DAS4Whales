"""
loc.py - Localisation module for the das4whales package.

This module provides functions for localizing the source of a sound source recorded by a DAS array.

Author: Quentin Goestchel, Léa Bouffaut
Date: 2024-06-18/2025-03-05
"""

import sys
import numpy as np
from spatial import calc_das_section_bearing, calc_source_position_lat_lon, calc_dist_lat_lon

def calc_arrival_times(t0, cable_pos, pos, c0):
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


def solve_lq(Ti, cable_pos, c0, Nbiter=10, fix_z=False, ninit=None, residuals=False, verbose=False):
    """
    Solve the least squares problem to localize the whale


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

    Returns
    -------
    n : np.ndarray
        Estimated whale position and time of emission vector [x, y, z, t0]

    """

    # Make a first guess of the whale position
    #TODO: make first guess a parameter
    n = np.array([40000, 1000, -30, np.min(Ti)])
    if ninit is not None:
        n = ninit

    # Regularization parameter
    lambda_reg = 1e-5

    for j in range(Nbiter):
        thj = calc_theta_vector(cable_pos, n)
        phij = calc_phi_vector(cable_pos, n)
        dt = Ti - calc_arrival_times(n[-1], cable_pos, n[:3], c0)

        # Fixed z case
        if fix_z:
            # Save z value to reappend it after the least squares computation
            dz = n[2]
            n_fz = np.delete(n, 2) # Remove z from the vector n
            del n # Delete n to reassign it with the new value
            n = n_fz # Reassign n without z

            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.ones_like(thj)]).T

        # Free z case
        else:
            # Compute the least squares coefficients matrix
            G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.sin(thj) / c0, np.ones_like(thj)]).T

        # Adding regularization to avoid singular matrix error
        lambda_identity = lambda_reg * np.eye(G.shape[1])

        dn = np.linalg.inv(G.T @ G + lambda_identity) @ G.T @ dt

        # Damping factor
        if j<4:
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


def calc_covariance_matrix(cable_pos, whale_pos, c0, var, fix_z=False):
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
        Variance of the residuals

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

    if np.linalg.cond(G.T @ G) > 1/sys.float_info.epsilon:
        print('Matrix is singular')
        lambda_reg = 1e-5
        lambda_identity = lambda_reg * np.eye(G.shape[1])
        cov = var * np.linalg.inv(G.T @ G + lambda_identity)
    else:
        cov = var * np.linalg.inv(G.T @ G)

    return cov


def calc_uncertainty_position(cable_pos, whale_pos, c0, var, fix_z=False):
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
        Variance of the residuals

    Returns
    -------
    unc : np.ndarray
        Uncertainties on the estimated whale position
    """

    cov = calc_covariance_matrix(cable_pos, whale_pos, c0, var, fix_z)
    unc = np.sqrt(np.diag(cov))

    return unc

