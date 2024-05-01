"""
improcess.py - image processing functions for DAS data

This module contains functions for image processing of DAS data.

Authors: Quentin Goestchel
Date: 2023-2024

"""

import numpy as np
import cv2
import scipy.signal as sp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.transform import radon, iradon
from tqdm import tqdm

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
    if dfx == 0:
        grad = -(image[:, :-dft] - image[:, dft:])
    elif dft == 0:
        grad = -(image[dfx:, :] - image[:-dfx, :])
    else:
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
    
    thresholded_output = (combined > threshold).float()
    
    return combined.squeeze(0)


def detect_long_lines(img):
    # Convert the image to grayscale (if it's not already)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy() * 255
    gray = np.uint8(gray)
    imglines = img.copy()

    plt.figure
    plt.imshow(gray, cmap='gray')
    plt.show()

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    plt.figure()
    plt.imshow(blurred, cmap='gray')
    plt.show()

    # Use Canny edge detection on the blurred image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=5, L2gradient=False)

    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.show()

    # Use Hough line transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=8)

    # Draw the lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return imglines


def bilateral_filter(img, diameter, sigma_color, sigma_space):
    # Apply bilateral filter to the image
    filtered_img = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return filtered_img


def compute_radon_transform(image, theta=None):
    # Compute the Radon transform
    radon_image = radon(image, theta=theta, circle=False)
    return radon_image


def gaussian_filter(img, size, sigma):
    # Apply Gaussian filter to the image
    filtered_img = cv2.GaussianBlur(img, (size, size), sigma)
    return filtered_img


def binning(image, ft, fx):
    """Apply binning to an image.

    This function applies binning to an image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    ft : float
        The binning factor along the time axis.
    fx : float
        The binning factor along the spatial axis.
    
    Returns
    -------
    numpy.ndarray
        The binned image.
    """

    img = cv2.resize(image, (0, 0), fx=ft, fy=fx, interpolation=cv2.INTER_AREA)
    return img

