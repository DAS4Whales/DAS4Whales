"""
improcess.py - image processing functions for DAS data

This module contains functions for image processing of DAS data.

Authors: Quentin Goestchel
Date: 2023-2024

"""

import numpy as np
import matplotlib.pyplot as plt

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
