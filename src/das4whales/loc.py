"""
loc.py - Localisation module for the das4whales package.

This module provides functions for localising the source of a sound source recorded by a DAS array.

Author: Quentin Goestchel
Date: 2024-06-18
"""

import numpy as np

def calc_arrival_times(t0, dist, pos, c0):
    """
    Get the theoretical arrival times of a whale call at a given distance along the cable
    """
    x, y, z = pos
    th_arrtimes = np.sqrt((dist - x) ** 2 + y ** 2 + z ** 2) / c0

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


def solve_lq(Ti, cable_pos, dist, c0, Nbiter=10):
    # Make a first guess of the whale position
    n = np.array([28000, 10, 800, np.min(Ti)])

    # Regularization parameter
    lambda_reg = 1e-5

    for j in range(Nbiter):
        thj = calc_theta_vector(cable_pos, n)
        phij = calc_phi_vector(cable_pos, n)
        dt = Ti - calc_arrival_times(n[-1], dist, n[:3], c0)
        # Compute the least squares coefficients matrix
        G = np.array([np.cos(thj) * np.cos(phij) / c0, np.cos(thj) * np.sin(phij) / c0, np.sin(thj) / c0, np.ones_like(thj)]).T

        # Adding regularization to avoid singular matrix error
        lambda_identity = lambda_reg * np.eye(G.shape[1])

        dn = np.linalg.inv(G.T @ G + lambda_identity) @ G.T @ dt

        if j<4:
            n += 0.7 * dn
        else:
            n += dn

        print(f'Iteration {j+1}: {n}')
        return n