"""
improcess.py - image processing functions for DAS data

This module contains functions for image processing of DAS data.

Authors: Quentin Goestchel
Date: 2023-2024

"""

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def gradient_oriented(image, direction):
    """Calculate the gradient oriented value of an image.

    This function calculates the gradient oriented value of an image based on the given direction.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    direction : tuple
        A tuple containing the direction values (dft, dfx).

    Returns
    -------
    numpy.ndarray
        The gradient oriented value of the image.

    """
    dft, dfx = direction
    grad = -(image[dfx:-dfx, :-dft] - 0.5 * image[2 * dfx:, dft:] - 0.5 * image[:-2*dfx, dft:])
    return grad


def detect_diagonal_edges(matrix, threshold):
    # Calculate derivatives along both axes
    # dx , dy = np.gradient(matrix)
    
    # Construct a diagonal filter kernel
    diagonal_filter = np.array([[2, -1, -1],
                                [-1, 2, -1],
                                [-1, -1, 2]])
    
    diagonal_filterleft = np.fliplr(diagonal_filter)

    # Convolve the gradient with the diagonal filter
    diagonal_gradient = np.abs(sp.fftconvolve(matrix, diagonal_filter, mode='same')) + np.abs(sp.fftconvolve(matrix, diagonal_filterleft, mode='same'))

    # Apply threshold to identify diagonal edges
    # diagonal_edges = diagonal_gradient > threshold

    return diagonal_gradient


def diagonal_edge_detection(img, threshold):
    """Detect diagonal edges in an image. 
    Inspired from https://github.com/Ocean-Data-Lab/das-finwhale-vocalization/

    This function detects diagonal edges in an image.

    Parameters

    ----------
    img : numpy.ndarray
        The input image.
    threshold : float
        The threshold value.

    Returns
    -------
    numpy.ndarray
        The diagonal edges in the image.

    """

    img = torch.tensor(img, dtype=torch.float32)


    weight_left = torch.tensor([[2, -1, -1], 
                                [-1, 2, -1],
                                [-1, -1, 2]], dtype=torch.float32)
    
    weight_right = torch.flip(weight_left, [0])
    
    conv_left = F.conv2d(img.unsqueeze(0), weight_left.unsqueeze(0).unsqueeze(0), padding=1)
    conv_right = F.conv2d(img.unsqueeze(0), weight_right.unsqueeze(0).unsqueeze(0), padding=1)
    
    combined = conv_left + conv_right
    
    sigmoid_output = torch.sigmoid(combined)
    
    thresholded_output = (combined > threshold).float()
    
    return thresholded_output.squeeze(0)