# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# # Attempt at associating the faint calls using line detection

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("reload_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %reload_ext autoreload
# %autoreload 2
# Imports   
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import das4whales as dw
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import matplotlib.cm as cm
import matplotlib.colors as colors
from tqdm import tqdm
import dask.array as da
# from dask import delayed
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit

# Load the peak indexes and the metadata
n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi2_th_4.nc') 
s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi2_th_4.nc')

# +
# Constants from the metadata

fs = n_ds.attrs['fs']
dx = n_ds.attrs['dx']
nnx = n_ds.attrs['data_shape'][0] + 1 #Fix this
snx = s_ds.attrs['data_shape'][0] #Fix this
n_selected_channels_m = n_ds.attrs['selected_channels_m']
s_selected_channels_m = s_ds.attrs['selected_channels_m']
    
# Constants management
c0 = 1480
n_selected_channels = dw.data_handle.get_selected_channels(n_selected_channels_m, dx)
s_selected_channels = dw.data_handle.get_selected_channels(s_selected_channels_m, dx)
n_begin_chan = n_selected_channels[0]
n_end_chan = n_selected_channels[1]
n_longi_offset = n_selected_channels[0] // n_selected_channels[2]
s_begin_chan = s_selected_channels[0]
s_end_chan = s_selected_channels[1]
s_longi_offset = s_selected_channels[0] // s_selected_channels[2]
n_dist = (np.arange(nnx) * n_selected_channels[2] + n_selected_channels[0]) * dx
s_dist = (np.arange(snx) * s_selected_channels[2] + s_selected_channels[0]) * dx
dx = dx * n_selected_channels[2]

# +
# load the peak indexes - North cable
npeakshf = n_ds["peaks_indexes_tp_HF"].values  # Extract as NumPy array
npeakslf = n_ds["peaks_indexes_tp_LF"].values
nSNRhf = n_ds["SNR_hf"].values
nSNRlf = n_ds["SNR_lf"].values

# load the peak indexes - South cable
speakshf = s_ds["peaks_indexes_tp_HF"].values
speakslf = s_ds["peaks_indexes_tp_LF"].values
sSNRhf = s_ds["SNR_hf"].values
sSNRlf = s_ds["SNR_lf"].values

# +
# Determine common color scale
vmin = min(np.min(nSNRhf), np.min(nSNRlf), np.min(sSNRhf), np.min(sSNRlf))
vmax = max(np.max(nSNRhf), np.max(nSNRlf), np.max(sSNRhf), np.max(sSNRlf))
cmap = cm.plasma  # Define colormap
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Normalize color range

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False)

# First subplot
sc1 = axes[0, 0].scatter(npeakshf[1][:] / fs, (n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3, 
                         c=nSNRhf, cmap=cmap, norm=norm, s=nSNRhf)
axes[0, 0].set_title('North Cable - HF')
axes[0, 0].set_ylabel('Distance [km]')
axes[0, 0].grid(linestyle='--', alpha=0.5)

# Second subplot
sc2 = axes[0, 1].scatter(npeakslf[1][:] / fs, (n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3, 
                         c=nSNRlf, cmap=cmap, norm=norm, s=nSNRlf)
axes[0, 1].set_title('North Cable - LF')
axes[0, 1].grid(linestyle='--', alpha=0.5)

# Third subplot
sc3 = axes[1, 0].scatter(speakshf[1][:] / fs, (s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3, 
                         c=sSNRhf, cmap=cmap, norm=norm, s=sSNRhf)
axes[1, 0].set_title('South Cable - HF')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Distance [km]')
axes[1, 0].grid(linestyle='--', alpha=0.5)

# Fourth subplot
sc4 = axes[1, 1].scatter(speakslf[1][:] / fs, (s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3, 
                         c=sSNRlf, cmap=cmap, norm=norm, s=sSNRlf)
axes[1, 1].set_title('South Cable - LF')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].grid(linestyle='--', alpha=0.5)

# Create a single colorbar for all subplots
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('SNR')

plt.show()

# -

# Choose to work on the HF or LF calls
n_peaks = npeakshf
s_peaks = speakshf
n_SNR = nSNRhf
s_SNR = sSNRhf
print(s_peaks.shape)

# ## Plot the map 

# +
# Import the cable location
df_north = pd.read_csv('../data/north_DAS_multicoord.csv')
df_south = pd.read_csv('../data/south_DAS_multicoord.csv')


# Extract the part of the dataframe used for the time picking process
idx_shift0 = int(n_begin_chan - df_north["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(n_end_chan - df_north["chan_idx"].iloc[-1])

df_north_used = df_north.iloc[idx_shift0:idx_shiftn:n_selected_channels[2]]

idx_shift0 = int(s_begin_chan - df_south["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(s_end_chan - df_south["chan_idx"].iloc[-1])

df_south_used = df_south.iloc[idx_shift0:idx_shiftn:s_selected_channels[2]]

# Import the bathymetry data
bathy, xlon, ylat = dw.map.load_bathymetry('../data/GMRT_OOI_RCA_Cables.grd')
print(f'Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

# Change the reference point to the last point
x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
print(xf, yf)
# # Create vectors of coordinates
utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
x = np.linspace(x0, xf, len(xlon))
y = np.linspace(y0, yf, len(ylat))

# -

dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)

# +

# Cable geometry (make it correspond to x,y,z = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2])
n_cable_pos = np.zeros((len(df_north_used), 3))
s_cable_pos = np.zeros((len(df_south_used), 3))

n_cable_pos[:, 0] = df_north_used['x']
n_cable_pos[:, 1] = df_north_used['y']
n_cable_pos[:, 2] = df_north_used['depth']

s_cable_pos[:, 0] = df_south_used['x']
s_cable_pos[:, 1] = df_south_used['y']
s_cable_pos[:, 2] = df_south_used['depth']

# print(n_cable_pos.shape)
# dist = np.arange(n_cable_pos.shape[0]) * (df_north['chan_m'][1] - df_north['chan_m'][0]) + df_north['chan_m'][idx_shift]

# +
from scipy.interpolate import RegularGridInterpolator

# Create a grid of coordinates, choosing the spacing of the grid
dx_grid = 2000 # [m]
dy_grid = 2000 # [m]
xg, yg = np.meshgrid(np.arange(xf, x0, dx_grid), np.arange(y0, yf, dy_grid))

ti = 0
zg = -40

interpolator = RegularGridInterpolator((x, y),  bathy.T)
bathy_interp = interpolator((xg, yg))

# Remove points if the ocean depth is too shallow (i.e., less than -25 m)
mask = bathy_interp < -25
# Compute arrival times only for valid grid points
# Flatten the grid points
xg, yg = xg[mask], yg[mask]

# In case of a meshgrid object (non flattened), use the following code:
# xg[~mask] = np.nan
# yg[~mask] = np.nan

# +
# Define KDE computation as a delayed function
# Define KDE computation as a delayed function
def compute_kde(delayed_picks, t_kde, bin_width, weights=None):
    """Computes the KDE of the delayed picks.

    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    bin_width : float
        Bin width for the KDE.

    Returns
    -------
    array-like
        KDE density values.  
    
    """
    if weights is not None:
        # Use weighted KDE, Scipy's gaussian_kde is faster that sklearn's KernelDensity for weighted KDE
        kde = gaussian_kde(delayed_picks, bw_method=bin_width / (np.max(t_kde) - np.min(t_kde)), weights=weights)
        density = kde(t_kde)
    else:
        kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
        log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        density = np.exp(log_dens) # Convert log-density to normal density
    return density


def compute_selected_picks(peaks, hyperbola, dt_sel, fs):
    """Selects picks that are closest to the hyperbola within a given time window."""
    selected_picks = ([], [])
    for i, idx in enumerate(peaks[1]):
        dist_idx = peaks[0][i]
        pick_time = idx / fs

        if hyperbola[dist_idx] - dt_sel < pick_time < hyperbola[dist_idx] + dt_sel:
            if dist_idx in selected_picks[0]:
                existing_idx = selected_picks[0].index(dist_idx)
                if abs(hyperbola[dist_idx] - pick_time) < abs(hyperbola[dist_idx] - selected_picks[1][existing_idx] / fs):
                    selected_picks[1][existing_idx] = idx  # Replace with closer pick
            else:
                selected_picks[0].append(dist_idx)
                selected_picks[1].append(idx)
    
    return np.array(selected_picks[0]), np.array(selected_picks[1])


def compute_curvature(w_times, w_distances):
    """Computes curvature using second derivatives."""
    ddx = np.diff(w_times)
    ddy = np.diff(w_distances)
    ddx2 = np.diff(ddx)
    ddy2 = np.diff(ddy)
    curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
    return np.mean(curvature)


# -

# Compute KDEs for all delayed picks
Nkde = 300
bin_width = 1
kde_hf = np.empty((xg.shape[0], Nkde))
shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
# Set the number of iterations for testing
iterations = 40

# Initialize the max_kde variable to enter the loop
n_up_peaks = np.copy(n_peaks)
s_up_peaks = np.copy(s_peaks)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)


# +
# Plot the arrival times for the grid
print(n_arr_tg.shape)
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('North Cable')
for i in range(xg.shape[0]):
            plt.plot(n_arr_tg[i, :], n_dist/1e3, ls='-', lw=1, color='tab:blue', alpha=0.1)

plt.subplot(1,2,2)
plt.title('South Cable')
for i in range(xg.shape[0]):
            plt.plot(s_arr_tg[i, :], s_dist/1e3, ls='-', lw=1, color='tab:blue', alpha=0.1)
plt.show()

# +

# Precompute the time indices
n_idx_times = np.array(n_up_peaks[1]) / fs # Update with the remaining peaks
s_idx_times = np.array(s_up_peaks[1]) / fs # Update with the remaining peaks

# Make a delayed picks array for all the grid points
# Broadcast the time indices delayed by the theoretical arrival times for the grid points
n_delayed_picks_hf = n_idx_times[None, :] - n_arr_tg[:, n_up_peaks[0]]
s_delayed_picks_hf = s_idx_times[None, :] - s_arr_tg[:, s_up_peaks[0]]

# Generate a time grid for each grid point by linearly spacing Nkde points 
# between the minimum and maximum delayed pick times. Transpose to ensure 
# the shape (shape_x, Nkde) (shape_x, Nkde) delayed_picks shape for consistency with KDE computation.
n_t_grid = np.linspace(np.min(n_delayed_picks_hf, axis=1), np.max(n_delayed_picks_hf, axis=1), Nkde).T
s_t_grid = np.linspace(np.min(s_delayed_picks_hf, axis=1), np.max(s_delayed_picks_hf, axis=1), Nkde).T

# Parallelized KDE computation
n_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(compute_kde)(n_delayed_picks_hf[i, :], n_t_grid[i, :], bin_width, weights=n_SNR) 
    for i in range(shape_x)
))

s_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(compute_kde)(s_delayed_picks_hf[i, :], s_t_grid[i, :], bin_width, weights=s_SNR)
    for i in range(shape_x)
))

# +
n_heatmap = np.max(n_kde_hf, axis=1)
s_heatmap = np.max(s_kde_hf, axis=1)

# Combined heatmap from summing the kdes
sum_kde_hf = n_kde_hf + s_kde_hf
# Hadamard product of the two kdes
prod_kde_hf = n_kde_hf * s_kde_hf

p_heatmap = np.max(prod_kde_hf, axis=1)
s_heatmap = np.max(s_kde_hf, axis=1)
# -

# ## Plot the Heatmap for the north cable

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green', label='Used cable locations')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points
ax.tricontourf(xg, yg, n_heatmap, levels=20, cmap='hot', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()
# -

# ## Plot the heatmap for the south cable

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points
ax.tricontourf(xg, yg, s_heatmap, levels=20, cmap='hot', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()
# -

# ## Plot the heatmaps for the North and South cables combined

# ### Sum of the maximums

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')
ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points

ax.tricontourf(xg, yg, n_heatmap + s_heatmap, levels=20, cmap='hot', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()
# -

# ### Maximum of the sum

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')
ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points

ax.tricontourf(xg, yg, s_heatmap, levels=20, cmap='hot', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()
# -

# #### Maximum of the product

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')
ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points

ax.tricontourf(xg, yg, p_heatmap, levels=20, cmap='hot', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()
# -

# ## Plot the heat map only for the points above the mean of the kde

# +
# Plot the grid points on the map
import cmocean.cm as cmo
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')
ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green')

# Plot the grid points
ax.scatter(xg, yg, c='k', s=1)

# Plot the heatmaps over the grid points

comb = n_heatmap / np.max(n_heatmap) * s_heatmap / np.max(s_heatmap)
comb[comb < np.mean(comb)] = 0
comb[comb > np.mean(comb)] = 1
ax.tricontourf(xg, yg, comb, levels=1, cmap='binary_r', alpha=0.5)

# Plot points along the cable every 10 km in terms of optical distance
# for i, point in enumerate(opticald_n, start=1):
#     # Plot the points
#     ax.plot(point[0], point[1], '.', color='k')
#     # Annotate the points with the distance
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=8)

# for i, point in enumerate(opticald_s, start=1):
#     ax.plot(point[0], point[1], '.', color='k')
#     ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -15), ha='center', fontsize=8)


# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('cable_Grid.pdf')
plt.show()

# +
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.colors import LightSource
# import matplotlib.tri as tri

# # Assuming these variables are already defined:
# # x, y - coordinate ranges
# # xg, yg - grid points
# # bathy - bathymetry data
# # extent - spatial extent for plotting
# # n_kde_hf, s_kde_hf - KDE arrays for north and south, shape (num_points, Nkde)
# # df_north, df_south - dataframes with cable coordinates
# # df_north_used, df_south_used - dataframes with used cable locations
# # Nkde - number of KDE frames
# # custom_cmap - your custom colormap

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.set_xlim(x[0], x[-1])
# ax.set_ylim(y[0], y[-1])
# ax.set_aspect('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_title('KDE Heatmap Animation')

# # Create LightSource for bathymetry relief
# ls = LightSource(azdeg=315, altdeg=45)
# rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, 
#                blend_mode='overlay', vmin=np.min(bathy), vmax=0)

# # Plot the bathymetry relief in background
# plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', 
#                  vmin=np.min(bathy), vmax=0)

# # Plot the cable location in 2D
# ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
# ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# # Plot the used cable locations
# ax.plot(df_south_used['x'], df_south_used['y'], 'tab:green', label='Used cable locations')
# ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green')

# # Plot the grid points
# sc = ax.scatter(xg, yg, c='k', s=1)

# # Add a legend
# ax.legend(loc='upper right')

# # norm kdes
# n_kde_hf = n_kde_hf / np.max(n_kde_hf)
# s_kde_hf = s_kde_hf / np.max(s_kde_hf)
# # Initialize contour
# contour = ax.tricontourf(xg, yg, n_kde_hf[:, 0] + s_kde_hf[:, 0], 
#                          levels=20, cmap='hot', alpha=0.5)

# # Update function
# def update(frame):
#     # Update contour
#     contour = ax.tricontourf(xg, yg, n_kde_hf[:, frame] + s_kde_hf[:, frame], 
#                              levels=20, cmap='hot', alpha=0.5)
#     return contour

# # Create animation

# ani = animation.FuncAnimation(fig, update, frames=Nkde, interval=100, blit=False)
# ani.save('kde_heatmap_animation.mp4', writer='ffmpeg', fps=10)
