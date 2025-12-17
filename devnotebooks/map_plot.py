# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize
import das4whales as dw
import scipy.ndimage
import cmocean


# Matplotlib settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 30

plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1.5


def map_plot():
    # Plot cable positions
    # Import the cable location
    df_north = pd.read_csv('./data/north_DAS_multicoord.csv')
    df_south = pd.read_csv('./data/south_DAS_multicoord.csv')

    # Import the bathymetry data
    bathy, xlon, ylat = dw.map.load_bathymetry('./data/GMRT_OOI_RCA_Cables.grd')
    print(f'Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

    # Plot the cables geometry in lat/lon coordinates
    # dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)
    # dw.map.plot_cables3D(df_north, df_south, bathy, xlon, ylat)

    # Convert the starting and ending points of the bathymetry grid to UTM coordinates
    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0]) # UTM of the first grid point
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1]) # UTM of the last grid point

    # Shift UTM coordinates so that the last bathymetry point becomes the origin (0,0)
    # This re-centers the coordinates relative to the final point
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0

    # x0, y0 = utm_x0, utm_y0
    # xf, yf = utm_xf, utm_yf

    # # Create vectors of coordinates
    utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
    utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
    x = np.linspace(x0, xf, len(xlon))
    y = np.linspace(y0, yf, len(ylat))

    # Plot the cables geometry in local coordinates
    # dw.map.plot_cables2D_m(df_north, df_south, bathy, x, y)
    # dw.map.plot_cables3D_m(df_north, df_south, bathy, x, y)

    # Convert Pacific City latitude and longitude to UTM coordinates 
    lat_pc, lon_pc = 45.201801, -123.960861 # Pacific City lat/lon
    utm_x_pc, utm_y_pc = dw.map.latlon_to_utm(lon_pc, lat_pc)

    # Adjust Pacific City's UTM coordinates to the shifted coordinate system (relative to utm_xf, utm_yf)
    utm_x_pc, utm_y_pc = utm_xf - utm_x_pc, utm_y_pc - utm_y0

    # Assuming `bathy` is your bathymetry data array
    bathy_smoothed = bathy.copy()

    # Create a mask for depths above 0 meters
    mask_above_zero = bathy > 0

    # Apply Gaussian smoothing only to values above 0 meters
    sigma = 20  # Adjust sigma for the level of smoothing desired
    smoothed_bathy = scipy.ndimage.gaussian_filter(bathy, sigma=sigma)

    # Combine smoothed and original data, preserving original values below 0 meters
    bathy_smoothed[mask_above_zero] = smoothed_bathy[mask_above_zero]


    # Plot the cables geometry in local coordinates, in 3D
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the bathymetry with light shading
    X, Y = np.meshgrid(x / 1e3, y / 1e3)  # Scale to kilometers if x and y are in meters
    rstride, cstride = X.shape[0] // 100, X.shape[1] // 150
  
    vmin = bathy_smoothed.min()
    vmax = bathy_smoothed.max()

    # Create a colormap normalization with explicit range and breaks at key values
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot the bathymetry with light shading, using the custom normalization
    surface = ax.plot_surface(X, Y, bathy_smoothed, rstride=rstride, cstride=cstride,
                            cmap=cmocean.cm.deep_r, norm=norm, antialiased=True, alpha=0.7)
    ax.invert_yaxis()

    depth_min = bathy_smoothed.min()
    depth_max = bathy_smoothed.max()

    # Add a vertical plane at y[0] to show the bathymetry profile
    # Extract the bathymetry profile along y[0]
    Z_plane = bathy_smoothed[0, :]

    # Define the X and Y coordinates for this plane
    X_plane = X[0, :]  # Take the X values along y[0]
    Y_plane = np.full_like(X_plane, Y[0, 0])  # Constant Y value along this line

    # Plot the bathymetry-matching vertical plane along y[0]
    ax.plot_surface(np.tile(X_plane, (2, 1)), np.tile(Y_plane, (2, 1)), np.vstack([np.full_like(Z_plane, depth_min), Z_plane]),
                    color='sandybrown', alpha=1, edgecolor='none', rasterized=True)

    # Add a vertical plane at x[0] to show the bathymetry profile
    # Extract the bathymetry profile along x[-1]
    Z_plane = bathy_smoothed[:, 0]

    # Define the X and Y coordinates for this plane
    Y_plane = Y[:, 0]  # Take the Y values along x[-1]
    X_plane = np.full_like(Y_plane, X[0, 0])  # Constant X value along this line

    # Plot the bathymetry-matching vertical plane along x[-1]
    ax.plot_surface(np.tile(X_plane, (2, 1)), np.tile(Y_plane, (2, 1)), np.vstack([np.full_like(Z_plane, depth_min), Z_plane]),
                    color='sandybrown', alpha=1, edgecolor='none')
    
    # Add contour lines for depth
    ax.contour(X, Y, bathy, levels=[-1400, -1200, -1000, -800, -600, -400, -300, -200, -100, 0], colors='k', linestyles='solid', linewidths=0.5)

    # Cables
    ax.plot(df_north['x'] / 1e3, df_north['y'] / 1e3, df_north['depth'], 'tab:red', label='North cable', lw=5, zorder=10)
    ax.plot(df_south['x'] / 1e3, df_south['y'] / 1e3, df_south['depth'], 'tab:orange', label='South cable', lw=5, zorder=10)

    # Pacific City
    ax.scatter(utm_x_pc / 1e3, utm_y_pc / 1e3, 10, color='black', s=1000, marker='*', zorder=10)
    ax.text(utm_x_pc / 1e3, utm_y_pc / 1e3, 100, "Pacific City", color='black', fontsize=30, ha='center', va='bottom', zorder=10)

    #1st repeaters 
    ax.scatter(df_north['x'].iloc[-1] / 1e3, df_north['y'].iloc[-1] / 1e3, df_north['depth'].iloc[-1], color='tab:red', s=300, label='1st repeater', marker='s', zorder=10)
    ax.scatter(df_south['x'].iloc[-1] / 1e3, df_south['y'].iloc[-1] / 1e3, df_south['depth'].iloc[-1], color='tab:orange', s=300, label='1st repeater', marker='s', zorder=10)

    # Labels and ticks
    ax.set_xlabel('x [km]', labelpad=30)
    ax.set_ylabel('y [km]', labelpad=35)
    ax.tick_params(axis='z', pad=30)
    ax.set_zlabel('Depth [m]', labelpad=50)

    # View angle
    ax.view_init(elev=40, azim=70)  # Elevation of 40 for a better perspective

    # Legend
    ax.set_aspect('equalxy')
    ax.legend(loc="upper right", frameon=True)
    plt.subplots_adjust(bottom=0.0, top=1, left=0.0, right=1)

    # Create a ScalarMappable for the colorbar with the same colormap and normalization
    mappable = cm.ScalarMappable(cmap=cmocean.cm.deep_r, norm=norm)
    mappable.set_array([])  # Needed for colorbar to work

    # Add the colorbar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=30, pad=0.001)
    cbar.set_label('Depth [m]')

    # plt.show()
    plt.savefig('test.png', dpi=300)

if __name__ == '__main__':
    map_plot()


