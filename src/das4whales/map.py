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
import cmocean.cm as cmo

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
    ds = xr.open_dataset(filepath, engine='scipy')
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
    # print(bathy.shape)
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
    
    # Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
    opticald_n = []
    opticald_s = []

    for i in range(int(10000/2-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
        opticald_n.append((df_north['lon'][i], df_north['lat'][i]))

    for i in range(int(10000/2-df_south["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
        opticald_s.append((df_south['lon'][i], df_south['lat'][i]))

    colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
    colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

    # Combine the color maps
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

    # Set extent of the plot
    extent = [xlon[0], xlon[-1], ylat[0], ylat[-1]]

    # Set the light source
    ls = LightSource(azdeg=350, altdeg=45)

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

    # Plot the cable location in 2D
    ax.plot(df_north['lon'], df_north['lat'], 'tab:red', label='North cable', lw=2.5)
    ax.plot(df_south['lon'], df_south['lat'], 'tab:orange', label='South cable', lw=2.5)

    # Add dashed contours at selected depths with annotations
    depth_levels = [-1500, -1000, -600, -250, -80]

    contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
    ax.clabel(contour_dashed, fmt='%d m', inline=True)

    # Plot points along the cable every 10 km in terms of optical distance
    for i, point in enumerate(opticald_n, start=1):
        # Plot the points
        ax.plot(point[0], point[1], '.', color='k')
        # Annotate the points with the distance
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 8), ha='center', fontsize=12)

    for i, point in enumerate(opticald_s, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=12)

    # Use a proxy artist for the color bar
    im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
    im_ratio = bathy.shape[1] / bathy.shape[0]
    plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0145)

    im.remove()

    # Set the labels
    plt.xlabel('Longitude [°]')
    plt.ylabel('Latitude [°]')
    plt.legend(loc='upper left')

    # Dashed grid lines
    plt.grid(linestyle='--', alpha=0.6, color='k')
    plt.tight_layout()
    plt.show()

    return


def plot_cables2D_m(df_north, df_south, bathy, xm, ym):
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
    xm : np.ndarray
        The x data vector in meters.
    ym : np.ndarray
        The y data vector in meters.
    """
    
    # Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
    opticald_n = []
    opticald_s = []

    for i in range(int(10000/2), len(df_north), int(10000/2)):
        opticald_n.append((df_north['x'][i], df_north['y'][i]))

    for i in range(int(10000/2), len(df_south), int(10000/2)):
        opticald_s.append((df_south['x'][i], df_south['y'][i]))

    # Chose a colormap to be sure that values above 0 are white, and values below 0 are blue
    colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
    colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

    # Combine the color maps
    all_colors = np.vstack((colors_undersea, colors_land))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
    extent = [xm[0], xm[-1], ym[0], ym[-1]]

    # Set the light source
    ls = LightSource(azdeg=350, altdeg=45)

    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)

    ax.plot(df_north['x'] , df_north['y'] , 'tab:red', label='North cable', lw=2.5)
    ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

    # Add dashed contours at selected depths with annotations
    depth_levels = [-1500, -1000, -600, -250, -80]

    contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
    ax.clabel(contour_dashed, fmt='%d m', inline=True)

    # Plot points along the cable every 10 km in terms of optical distance
    for i, point in enumerate(opticald_n, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 8), ha='center', fontsize=12)

    for i, point in enumerate(opticald_s, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=12)

    # Use a proxy artist for the color bar
    im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
    im_ratio = bathy.shape[1] / bathy.shape[0]
    plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)

    im.remove()

    plt.subplots_adjust(bottom=0.0, top=1, left=0.0, right=1)
    
    # Set the labels
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    plt.legend(loc='upper left')
    plt.grid(linestyle='--', alpha=0.6, color='k')
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

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the bathymetry
    X, Y = np.meshgrid(xlon, ylat)

    # Set the stride of the plot by dividing the number of points by 100 in the x direction and 50 in the y direction
    rstride = X.shape[0] // 100
    cstride = X.shape[1] // 50

    # print(rstride, cstride)
    # Plot the surface
    ax.plot_surface(X, Y, bathy, cmap='Blues_r', alpha=0.7, antialiased=True, rstride=rstride, cstride=cstride)
    # Plot the cables
    ax.plot(df_north['lon'], df_north['lat'], df_north['depth'], 'tab:red', label='North cable', lw=4)
    ax.plot(df_south['lon'], df_south['lat'], df_south['depth'], 'tab:orange', label='South cable', lw=4)
    # Set labels distance to the axis
    ax.set_xlabel('Longitude', labelpad=30)
    ax.set_ylabel('Latitude', labelpad=30)
    ax.set_zlabel('Depth [m]', labelpad=35)

    # Set the distance between tick labels and axis
    ax.tick_params(axis='x', pad=10)  # Adjust X-axis tick label distance
    ax.tick_params(axis='y', pad=10)  # Adjust Y-axis tick label distance
    ax.tick_params(axis='z', pad=20)  # Adjust Z-axis tick label distance
    # Set the angle of view
    ax.view_init(elev=40, azim=250)
    ax.set_aspect('equalxy')
    ax.legend()
    plt.show()
    plt.close()

    return


def plot_cables3D_m(df_north, df_south, bathy, x, y):
    """
    Plot the cables on the bathymetry map in 3D.

    Parameters
    ----------
    df_north : pandas.DataFrame
        The dataframe containing the north cable coordinates.
    df_south : pandas.DataFrame
        The dataframe containing the south cable coordinates in meters.
    bathy : np.ndarray
        The bathymetry data array. zij = bathy[i,j] is the depth at the point (xlon[j], ylat[i]).
    x : np.ndarray
        The x data vector.
    y : np.ndarray
        The y data vector.
    """

    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the bathymetry
    X, Y = np.meshgrid(x / 1e3, y / 1e3) 

    # Set the stride of the plot by dividing the number of points by 100 in the x direction and 50 in the y direction
    rstride = X.shape[0] // 100
    cstride = X.shape[1] // 50

    print(rstride, cstride)
    # Plot the surface
    ax.plot_surface(X, Y, bathy, cmap='Blues_r', alpha=0.7, antialiased=True, rstride=rstride, cstride=cstride)
    # Plot the cables
    ax.plot(df_north['x'] / 1e3, df_north['y'] / 1e3, df_north['depth'], 'tab:red', label='North cable', lw=4)
    ax.plot(df_south['x'] / 1e3, df_south['y'] / 1e3, df_south['depth'], 'tab:orange', label='South cable', lw=4)

    ax.invert_yaxis()

    # Set labels distance to the axis
    ax.set_xlabel('x [km]', labelpad=30)
    ax.set_ylabel('y [km]', labelpad=35)
    ax.set_zlabel('Depth [m]', labelpad=35)

    # Set the distance between tick labels and axis
    ax.tick_params(axis='x')  # Adjust X-axis tick label distance
    ax.tick_params(axis='y')  # Adjust Y-axis tick label distance
    ax.tick_params(axis='z', pad=20)  # Adjust Z-axis tick label distance
    # Set the angle of view
    ax.view_init(elev=40, azim=70)

    ax.set_aspect('equalxy')
    ax.legend()
    plt.subplots_adjust(bottom=0.0, top=1, left=0.0, right=1)
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

def latlon_to_xy(lat, lon, lat_ref=None, lon_ref=None, alt=0):
    # This is simplified conversion, assuming earth's curvature is spherical at the scale of interest.
    # It uses the estimated radius of the earth using WGS-84 ellipsoid flattening, and converts
    # lat/lon differences to meters assuming a sphere with that radius.
    
    if lat_ref is None:
        lat_ref = lat.mean()
    
    if lon_ref is None:
        lon_ref = lon.mean()
    
    a = 6378137.0  # WGS-84 equatorial radius in meters
    f = 1 / 298.257223563  # WGS-84 flattening factor
    e2 = f * (2 - f)  # Square of eccentricity
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    N = a / np.sqrt(1 - e2 * np.sin(lat_ref_rad)**2)
    M = a * (1 - e2) / (1 - e2 * np.sin(lat_ref_rad)**2)**1.5
    d_lat = lat_rad - lat_ref_rad
    d_lon = lon_rad - lon_ref_rad
    x = np.array(d_lon * (N + alt) * np.cos(lat_ref_rad))
    y = np.array(d_lat * (M + alt))
    return x, y