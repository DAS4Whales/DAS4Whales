"""
loc.py - Localisation module for the das4whales package.

This module provides functions for localizing the source of a sound source recorded by a DAS array.

Author: Quentin Goestchel
Date: 2024-06-18
"""

import sys
import numpy as np

def calc_arrival_times(t0, cable_pos, pos, c0):
    """
    Get the theoretical arrival times of a whale call at a given distance along the cable
    """
    # Whale position or potential whale position
    x, y, z = pos

    # Cable positions for each channel
    x_cable, y_cable, z_cable = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2]

    th_arrtimes = np.sqrt((x_cable - x) ** 2 + (y_cable - y) ** 2 + (z_cable - z) ** 2) / c0

    return t0 + th_arrtimes


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


def solve_lq(Ti, cable_pos, c0, Nbiter=10, fix_z=False):
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
    n = np.array([300000, 5000000-2000, 300, np.min(Ti)])

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

        if j<4:
            n += 0.7 * dn
        else:
            n += dn

        if fix_z:
            # reappend z to n in index 2 (before index 3)
            n = np.insert(n, 2, dz)            

        print(f'Iteration {j+1}: x = {n[0]:.4f} m, y = {n[1]:.4f}, z = {n[2]:.4f}, ti = {n[3]:.4f}')

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