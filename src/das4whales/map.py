"""
map.py - Map creation module for the das4whales package.

This module provides functions for coordinates handling and map visualization for DAS data.

Author: Quentin Goestchel
Date: 2024-06-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
import pandas as pd
import xarray as xr
import pyproj

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


def load_bathymetry(filepath):
    """
    Load the bathymetry data from a text file.

    Parameters
    ----------
    filepath : str
        The file path to the bathymetry data file. '.grd' file format is used here and can be found at https://www.gmrt.org/GMRTMapTool/. 

    Returns
    -------
    bathy : np.ndarray
        The bathymetry data array. zij = bathy[i,j] is the depth at the point (xlon[j], ylat[i]).
    xlon : np.ndarray
        The longitude data vector.
    ylat : np.ndarray
        The latitude data vector.
    """

    # Import the bathymetry data
    ds = xr.open_dataset('data/GMRT_OOI_RCA_Cables.grd')
    # Extract the bathymetry values
    bathy = ds['z'].values

    if np.isnan(bathy).any():
        print("NaNs detected in the dataset.")

    # Extract the dimensions
    dim = np.flip(ds.dimension).values
    # Reshape the bathymetry
    bathy = bathy.reshape(dim)
    # Flip the bathymetry
    bathy = np.flipud(bathy)

    # Remove columns and rows with NaN values
    bathy = bathy[~np.isnan(bathy).all(axis=1)]
    bathy = bathy[:, ~np.isnan(bathy).all(axis=0)]

    # Extract the x and y ranges
    x0, xf = ds['x_range'].values
    y0, yf = ds['y_range'].values
    print(f'latitude longitude span: x0 = {x0}, xf = {xf}, y0 = {y0}, yf = {yf}')

    # Create the x and y coordinates vectors
    print(bathy.shape)
    dim = bathy.shape
    xlon = np.linspace(x0, xf, dim[1])
    ylat = np.linspace(y0, yf, dim[0])

    return bathy, xlon, ylat


def flatten_bathy(bathy, threshold):
    """
    Flatten the bathymetry above a certain threshold.

    Parameters
    ----------
    bathy : np.ndarray
        The bathymetry data array. zij = bathy[i,j] is the depth at the point (xlon[j], ylat[i]).
    threshold : float
        The threshold above which the bathymetry is flattened.

    Returns
    -------
    bathy_flat : np.ndarray
        The flattened bathymetry data array.
    """
    # Copy the bathymetry array
    bathy_flat = bathy.copy()
    # Flatten the bathymetry above the threshold value and assign the threshold value to the rest
    bathy_flat[bathy_flat > threshold] = threshold

    return bathy_flat


def plot_cables2D(df_north, df_south, bathy, xlon, ylat):
    """
    Plot the cables on the bathymetry map.

    Parameters
    ----------
    df_north : pandas.DataFrame
        The dataframe containing the north cable coordinates.
    df_south : pandas.DataFrame
        The dataframe containing the south cable coordinates.
    bathy : np.ndarray
        The bathymetry data array. zij = bathy[i,j] is the depth at the point (xlon[j], ylat[i]).
    xlon : np.ndarray
        The longitude data vector.
    ylat : np.ndarray
        The latitude data vector.
    """
    
    # Chose a colormap to be sure that values above 0 are white, and values below 0 are blue
    colors_undersea = plt.cm.Blues_r(np.linspace(0, 0.5, 100)) # blue colors for under the sea
    colors_land = np.array([[1, 1, 1, 1]] * 40)  # white for above zero

    # Combine the color maps
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
    extent = [min(xlon), max(xlon), min(ylat), max(ylat)]

    # Set the light source
    ls = LightSource(azdeg=350, altdeg=45)

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay')
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower')

    # TODO: Make this less ugly
    # Check if df_north is a dataframe
    if not isinstance(df_north, pd.DataFrame):
        # Plot the cable location in 2D
        ax.plot(df_north[0], df_north[1], 'tab:red', label='North cable')
        ax.plot(df_south[0], df_south[1], 'tab:orange', label='South cable')
    else:
        # Plot the cable location in 2D
        ax.plot(df_north['lon'], df_north['lat'], 'tab:red', label='North cable')
        ax.plot(df_south['lon'], df_south['lat'], 'tab:orange', label='South cable')
        # plt.plot(xlon[0], ylat[-1], 'o', color='tab:red', label='test' )

    # Draw isoline at 0
    ax.contour(bathy, levels=[0], colors='k', extent=extent)

    # Use a proxy artist for the color bar
    im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower')
    plt.colorbar(im, ax=ax, label='Depth [m]', aspect=50, pad=0.1, orientation='horizontal')
    im.remove()

    if not isinstance(df_north, pd.DataFrame):
        # Set the labels
        plt.xlabel('UTM x [m]')
        plt.ylabel('UTM y [m]')

    else:
        # Set the labels
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()

    return


def plot_cables3D(df_north, df_south, bathy, xlon, ylat):
    """
    Plot the cables on the bathymetry map in 3D.

    Parameters
    ----------
    df_north : pandas.DataFrame
        The dataframe containing the north cable coordinates.
    df_south : pandas.DataFrame
        The dataframe containing the south cable coordinates.
    bathy : np.ndarray
        The bathymetry data array. zij = bathy[i,j] is the depth at the point (xlon[j], ylat[i]).
    xlon : np.ndarray
        The longitude data vector.
    ylat : np.ndarray
        The latitude data vector.
    """

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the bathymetry
    X, Y = np.meshgrid(xlon, ylat)

    # Set the stride of the plot by dividing the number of points by 100 in the x direction and 50 in the y direction
    rstride = X.shape[0] // 100
    cstride = X.shape[1] // 50

    print(rstride, cstride)
    # Plot the surface
    ax.plot_surface(X, Y, bathy, cmap='Blues_r', alpha=0.7, antialiased=True, rstride=rstride, cstride=cstride)
    # Plot the cables
    ax.plot(df_north['lon'], df_north['lat'], df_north['depth'], 'tab:red', label='North cable', lw=4)
    ax.plot(df_south['lon'], df_south['lat'], df_south['depth'], 'tab:orange', label='South cable', lw=4)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth [m]')
    ax.set_aspect('equalxy')
    ax.legend()
    plt.show()

    return


def latlon_to_utm(lon, lat, zone=10):
    """
    Convert latitude and longitude to UTM coordinates for a specified zone

    Parameters
    ----------
    lon : float
        The longitude.
    lat : float
        The latitude.
    zone : int
        The UTM zone.

    Returns
    -------
    utm_x : float
        The UTM x coordinate.
    utm_y : float
        The UTM y coordinate.
    """

    # Define the WGS84 coordinate system and the UTM coordinate system for the specified zone
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_zone = pyproj.CRS(f"EPSG:326{zone:02d}")

    # Create a transformer object to convert from WGS84 to UTM
    transformer = pyproj.Transformer.from_crs(wgs84, utm_zone, always_xy=True)

    # Perform the transformation
    utm_x, utm_y = transformer.transform(lon, lat)

    return utm_x, utm_y