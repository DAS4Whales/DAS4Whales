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

def ScalePixels(img):
    img = (img - img.min())/(img.max() - img.min())
    return img


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
    # Sobel adaptation
    # diagonal_filter = np.array([[ 0,  1,  2],
    #                             [-1,  0,  1],
    #                             [-2, -1,  0]])
    
    # Prewitt adaptation
    # diagonal_filter = np.array([[ 0,  1,  1],
    #                             [-1,  0,  1],
    #                             [-1, -1,  0]])
    
    #Second order
    # diagonal_filter = np.array([[-1, -1, 2], 
    #                             [-1, 2, -1],
    #                             [2, -1, -1]])
    
    diagonal_filter = np.array([[ 0,  1,  1,  1,  1],
                                [-1,  0,  1,  1,  1],
                                [-1, -1, 0,   1,  1],
                                [-1, -1, -1,  0,  1],
                                [-1, -1, -1, -1,  0]])
    
    # diagonal_filter = np.array([[ 0,  1,  1,  1,  1,  1,  1],
    #                             [-1,  0,  1,  1,  1,  1,  1],
    #                             [-1, -1,  0,  1,  1,  1,  1],
    #                             [-1, -1, -1,  0,  1,  1,  1],
    #                             [-1, -1, -1, -1,  0,  1,  1],
    #                             [-1, -1, -1, -1, -1,  0,  1],
    #                             [-1, -1, -1, -1, -1, -1,  0]])
    
    # diagonal_filter = np.array([[ 0,  1,  1,  1,  1,  1,  1,  1],
    #                             [-1,  0,  1,  1,  1,  1,  1,  1],
    #                             [-1, -1,  0,  1,  1,  1,  1,  1],
    #                             [-1, -1, -1,  0,  1,  1,  1,  1],
    #                             [-1, -1, -1, -1,  0,  1,  1,  1],
    #                             [-1, -1, -1, -1, -1,  0,  1,  1],
    #                             [-1, -1, -1, -1, -1, -1,  0,  1],
    #                             [-1, -1, -1, -1, -1, -1, -1,  0]])


    diagonal_filterleft = np.fliplr(diagonal_filter)

    # Convolve the gradient with the diagonal filter
    diagonal_gradient = sp.fftconvolve(matrix, diagonal_filter, mode='same') + sp.fftconvolve(matrix, diagonal_filterleft, mode='same')

    # diagonal_gradient /= np.sum(np.abs(diagonal_gradient))

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
    
    thresholded_output = (sigmoid_output > threshold).float()
    
    return thresholded_output.squeeze(0)



def generate_directional_kernel(angle):
    radians = np.deg2rad(angle)
    kernel_size = 3  # Size of the kernel
    half_size = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    center = half_size

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            # Calculate the angle of the current position relative to the center
            position_angle = np.arctan2(y, x)
            # Calculate the difference between the position angle and the desired angle
            angle_diff = np.abs(position_angle - radians)
            # Ensure the angle difference is within the range of 0 to pi/2
            angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
            # Set the value of the kernel based on the angle difference
            kernel[i, j] = np.cos(angle_diff)

    return kernel