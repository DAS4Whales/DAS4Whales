"""
map.py - Map creation module for the das4whales package.

This module provides functions for coordinates handling and map visualization for DAS data.

Author: Quentin Goestchel
Date: 2024-06-26
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd

def load_cable_coordinates(filepath, dx):
    """
    Load the cable coordinates from a text file.

    Parameters
    ----------
    filepath : str
        The file path to the cable coordinates file.
    dx : float
        The distance between two channels.

    Returns
    -------
    df : pandas.DataFrame
        The cable coordinates dataframe.
    """

    # load the .txt file and create a pandas dataframe
    df = pd.read_csv(filepath, delimiter = ",", header = None)
    df.columns = ['chan_idx','lat', 'lon', 'depth']
    df['chan_m'] = df['chan_idx'] * dx

    return df