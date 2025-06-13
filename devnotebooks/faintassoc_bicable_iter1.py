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
import scipy.signal as sp
plt.rcParams['font.size'] = 30
plt.rcParams['lines.linewidth'] = 3

# Load the peak indexes and the metadata
n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

# +
# Constants from the metadata

fs = n_ds.attrs['fs']
dx = n_ds.attrs['dx']
nnx = n_ds.attrs['data_shape'][0]
snx = s_ds.attrs['data_shape'][0]
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
fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False, constrained_layout=True)

# First subplot
sc1 = axes[0, 0].scatter(npeakshf[1][:] / fs, (n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3, 
                         c=nSNRhf, cmap=cmap, norm=norm, s=nSNRhf, rasterized=True)
axes[0, 0].set_title('North Cable - HF')
axes[0, 0].set_ylabel('Distance [km]')
axes[0, 0].grid(linestyle='--', alpha=0.5)
axes[0, 0].set_xlim(min(npeakshf[1][:] / fs), max(npeakshf[1][:] / fs))
axes[0, 0].set_ylim(min(n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3, 
                       max(n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3)
# Set the x-axis ticks and labels

# Second subplot
sc2 = axes[0, 1].scatter(npeakslf[1][:] / fs, (n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3, 
                         c=nSNRlf, cmap=cmap, norm=norm, s=nSNRlf, rasterized=True)
axes[0, 1].set_title('North Cable - LF')
axes[0, 1].grid(linestyle='--', alpha=0.5)
axes[0, 1].set_yticklabels([])
axes[0, 1].set_xlim(min(npeakslf[1][:] / fs), max(npeakslf[1][:] / fs))
axes[0, 1].set_ylim(min(n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3,
                       max(n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3)

# Third subplot
sc3 = axes[1, 0].scatter(speakshf[1][:] / fs, (s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3, 
                         c=sSNRhf, cmap=cmap, norm=norm, s=sSNRhf, rasterized=True)
axes[1, 0].set_title('South Cable - HF')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Distance [km]')
axes[1, 0].grid(linestyle='--', alpha=0.5)
axes[1, 0].set_xlim(min(speakshf[1][:] / fs), max(speakshf[1][:] / fs))
axes[1, 0].set_ylim(min(s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3,
                       max(s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3)
# Set the x-axis ticks and labels
axes[1, 0].set_xticks(np.arange(0, max(npeakshf[1][:] / fs), 10))


# Fourth subplot
sc4 = axes[1, 1].scatter(speakslf[1][:] / fs, (s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3, 
                         c=sSNRlf, cmap=cmap, norm=norm, s=sSNRlf, rasterized=True)
axes[1, 1].set_title('South Cable - LF')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].grid(linestyle='--', alpha=0.5)
axes[1, 1].set_yticklabels([])
axes[1, 1].set_xlim(min(speakslf[1][:] / fs), max(speakslf[1][:] / fs))
axes[1, 1].set_ylim(min(s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3,
                       max(s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3)
# Set the x-axis ticks and labels
axes[1, 1].set_xticks(np.arange(0, max(speakslf[1][:] / fs), 10))

# Create a single colorbar for all subplots
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('SNR Estimation [dB]')
# adapt the hspace between the subplots
plt.savefig('../figs/Sparse_peaks.pdf', bbox_inches='tight', transparent=True, format='pdf')
plt.show()

# -

# ## Plot the map 

# +
# Import the cable location
df_north = pd.read_csv('../data/north_DAS_multicoord.csv')
df_south = pd.read_csv('../data/south_DAS_multicoord.csv')


# Extract the part of the dataframe used for the time picking process
idx_shift0 = int(n_begin_chan - df_north["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(n_end_chan - df_north["chan_idx"].iloc[-1])

df_north_used = df_north.iloc[idx_shift0:idx_shiftn:n_selected_channels[2]][:nnx]

idx_shift0 = int(s_begin_chan - df_south["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(s_end_chan - df_south["chan_idx"].iloc[-1])

df_south_used = df_south.iloc[idx_shift0:idx_shiftn:s_selected_channels[2]][:snx]

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
        kde = gaussian_kde(delayed_picks, bw_method=bin_width/np.std(delayed_picks), weights=weights)
        density = kde(t_kde)
        # kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        # kde.fit(delayed_picks[:, None], sample_weight=weights) # Reshape to (n_samples, 1)
        # log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        # density = np.exp(log_dens) # Convert log-density to normal density
    else:
        kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
        log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        density = np.exp(log_dens) # Convert log-density to normal density
    return density


def fast_kde_rect(delayed_picks, t_kde, overlap=None, bin_width=None, weights=None):
    """
    Fast KDE approximation using histogram and optional rectangular smoothing.
    
    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    """
    # Histogram the picks
    hist_range = (t_kde[0], t_kde[-1])
    bins = len(t_kde)
    
    hist, _ = np.histogram(delayed_picks, bins=bins, range=hist_range, weights=weights)
    
    # Optional rectangular smoothing
    if overlap is None:
        overlap = np.diff(t_kde).mean()
    if bin_width is None:
        bin_width = 2 * overlap
    kernel_bins = int(np.round(bin_width / overlap))
    if kernel_bins % 2 == 0:
        kernel_bins += 1  # Ensure odd length
    kernel = np.ones(kernel_bins) / kernel_bins
    hist = sp.convolve(hist, kernel, mode="same")
    
    return hist / np.trapezoid(hist, t_kde)  # Normalize to match KDE style



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
# TODO: KDE number of points proportional to the number of picks (y-axis)?
dt_kde = 0.5 # [s] Time resolution of the KDE (overlap)
bin_width = 1
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
# Set the number of iterations for testing
iterations = 40

# +
# Initialize the max_kde variable to enter the loop
n_associated_list = []
n_used_hyperbolas = []
n_rejected_list = []
n_rejected_hyperbolas = []

s_associated_list = []
s_used_hyperbolas = []
s_rejected_list = []
s_rejected_hyperbolas = []

n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# n_arr_tg -= np.min(n_arr_tg, axis=1, keepdims=True)
# s_arr_tg -= np.min(s_arr_tg, axis=1, keepdims=True)

print(n_arr_tg.shape, nnx, n_cable_pos.shape)
print(s_arr_tg.shape, snx, s_cable_pos.shape)

# n_arr_tg = n_arr_tg[np.min(n_arr_tg, axis=1) > 20]

# +
# Plot the arrival times for the grid
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('North Cable')
for i in range(xg.shape[0]):
            plt.plot(n_arr_tg[i, :], n_dist/1e3, ls='-', lw=1, color='tab:blue', alpha=0.1)
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.xlabel('Time [s]')
# Remove the upper part of the bounding box 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.title('South Cable')
for i in range(xg.shape[0]):
            plt.plot(s_arr_tg[i, :], s_dist/1e3, ls='-', lw=1, color='tab:blue', alpha=0.1)
plt.xlabel('Time [s]')
# Remove the upper part of the bounding box 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(linestyle='--', alpha=0.5)

plt.savefig('../figs/toa.pdf', bbox_inches='tight')
plt.show()
# -

print(xg.shape)

# +
# Plot the arrival times for the grid
cmap = plt.get_cmap('plasma')
examples= [6, 480, 700] # Example indices to plot
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('North Cable')
for i in range(xg.shape[0]):
    plt.plot(n_arr_tg[i, :], n_dist/1e3, ls='-', lw=0.5, color='tab:blue', alpha=0.1)
for i in examples:
    color = cmap(i / xg.shape[0]) 
    plt.plot(n_arr_tg[i, :], n_dist/1e3, ls='-', lw=2, color=color)

plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.xlabel('Time [s]')
# Remove the upper part of the bounding box 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.title('South Cable')
for i in range(xg.shape[0]):
            plt.plot(s_arr_tg[i, :], s_dist/1e3, ls='-', lw=0.5, color='tab:blue', alpha=0.1)

for i in examples:
    color = cmap(i / xg.shape[0]) 
    plt.plot(s_arr_tg[i, :], s_dist/1e3, ls='-', lw=2, color=color)

plt.xlabel('Time [s]')
# Remove the upper part of the bounding box 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(linestyle='--', alpha=0.5)

plt.savefig('../figs/toa.pdf', bbox_inches='tight')
plt.show()

# +
n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs # Update with the remaining peaks
n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs # Update with the remaining peaks
s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs # Update with the remaining peaks
s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs # Update with the remaining peaks

# Make a delayed picks array for all the grid points
# Broadcast the time indices delayed by the theoretical arrival times for the grid points

n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

global_min = min(np.min(n_delayed_picks_hf), np.min(n_delayed_picks_lf), np.min(s_delayed_picks_hf), np.min(s_delayed_picks_lf))
global_max = max(np.max(n_delayed_picks_hf), np.max(n_delayed_picks_lf), np.max(s_delayed_picks_hf), np.max(s_delayed_picks_lf))
Nkde=np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
t_kde = np.linspace(global_min, global_max, Nkde)

print(Nkde)


# +
# n_kde_hf = np.array(Parallel(n_jobs=-1)(
#     delayed(compute_kde)(n_delayed_picks_hf[i, :], t_kde, bin_width, weights=nSNRhf) 
#     for i in range(n_shape_x)
# ))

# n_kde_lf = np.array(Parallel(n_jobs=-1)(
#     delayed(compute_kde)(n_delayed_picks_lf[i, :], t_kde, bin_width, weights=nSNRlf)
#     for i in range(n_shape_x)
# ))


# s_kde_hf = np.array(Parallel(n_jobs=-1)(
#     delayed(compute_kde)(s_delayed_picks_hf[i, :], t_kde, bin_width, weights=sSNRhf)
#     for i in range(s_shape_x)
# ))

# s_kde_lf = np.array(Parallel(n_jobs=-1)(
#     delayed(compute_kde)(s_delayed_picks_lf[i, :], t_kde, bin_width, weights=sSNRlf)
#     for i in range(s_shape_x)
# ))
# time with topha: 4m55s
# -

n_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(fast_kde_rect)(n_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRhf) 
    for i in range(n_shape_x)
))
n_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(fast_kde_rect)(n_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRlf)
    for i in range(n_shape_x)
))
s_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(fast_kde_rect)(s_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRhf)
    for i in range(s_shape_x)
))
s_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(fast_kde_rect)(s_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRlf)
    for i in range(s_shape_x)
))

print(n_kde_hf.shape, n_kde_lf.shape)
print(n_delayed_picks_hf.shape, n_delayed_picks_lf.shape)
print(s_kde_hf.shape, s_kde_lf.shape)
print(s_delayed_picks_hf.shape, s_delayed_picks_lf.shape)

hf_kde = n_kde_hf + s_kde_hf
lf_kde = n_kde_lf + s_kde_lf

# +
# Find the maximum for the 4 kde sets 

n_max_kde_hf = np.argmax(n_kde_hf)
nhf_imax, nhf_tmax = np.unravel_index(n_max_kde_hf, n_kde_hf.shape)

n_max_kde_lf = np.argmax(n_kde_lf)
nlf_imax, nlf_tmax = np.unravel_index(n_max_kde_lf, n_kde_lf.shape)

s_max_kde_hf = np.argmax(s_kde_hf)
shf_imax, shf_tmax = np.unravel_index(s_max_kde_hf, s_kde_hf.shape)

s_max_kde_lf = np.argmax(s_kde_lf)
slf_imax, slf_tmax = np.unravel_index(s_max_kde_lf, s_kde_lf.shape)

print(f'North HF max kde: {n_max_kde_hf}, max index: {nhf_imax}, max time: {nhf_tmax}')
print(f'North LF max kde: {n_max_kde_lf}, max index: {nlf_imax}, max time: {nlf_tmax}')
print(f'South HF max kde: {s_max_kde_hf}, max index: {shf_imax}, max time: {shf_tmax}')
print(f'South LF max kde: {s_max_kde_lf}, max index: {slf_imax}, max time: {slf_tmax}')

# Find the maximum for the 2 combined kde sets
hf_max_kde = np.argmax(hf_kde)
hf_imax, hf_tmax = np.unravel_index(hf_max_kde, hf_kde.shape)

lf_max_kde = np.argmax(lf_kde)
lf_imax, lf_tmax = np.unravel_index(lf_max_kde, lf_kde.shape)

print(f'Combined HF max kde: {hf_max_kde}, max index: {hf_imax}, max time: {hf_tmax}')
print(f'Combined LF max kde: {lf_max_kde}, max index: {lf_imax}, max time: {lf_tmax}')


# +
# Plot the KDE
plt.figure(figsize=(20,12))
plt.subplot(3,1,1)
plt.title('North Cable')
# plt.plot(n_t_grid_hf[nhf_imax, :], n_kde_hf[nhf_imax, :], color='tab:blue', lw=2, label='north HF')
# plt.plot(n_t_grid_lf[nlf_imax, :], n_kde_lf[nlf_imax, :], color='tab:orange', lw=2, label='north LF')
plt.plot(t_kde, n_kde_hf[nhf_imax, :], color='tab:blue', lw=2, label='north HF')
plt.plot(t_kde, n_kde_lf[nlf_imax, :], color='tab:orange', lw=2, label='north LF')
plt.plot(t_kde, n_kde_hf[hf_imax, :], color='tab:cyan', lw=2, ls='--', label='north HF, maxsingle')
plt.plot(t_kde, n_kde_lf[lf_imax, :], color='tab:red', lw=2, ls='--', label='north LF, maxsingle')
plt.xlim(0, 60)
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.ylabel('Probability density [-]')

plt.grid(linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3,1,2)
plt.title('South Cable')
plt.plot(t_kde, s_kde_hf[shf_imax, :], color='tab:blue', lw=2, label='south HF')
plt.plot(t_kde, s_kde_lf[slf_imax, :], color='tab:orange', lw=2, label='south LF')
plt.plot(t_kde, s_kde_hf[hf_imax, :], color='tab:cyan', lw=2, ls='--', label='south HF, maxsingle')
plt.plot(t_kde, s_kde_lf[lf_imax, :], color='tab:red', lw=2, ls='--', label='south LF, maxsingle')
plt.ylabel('Probability density [-]')
plt.xlim(0, 60)
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.grid(linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3,1,3)
plt.title('Combined KDE')
plt.plot(t_kde, hf_kde[hf_imax, :], color='tab:green', lw=2, label="Combined HF")
plt.plot(t_kde, lf_kde[lf_imax, :], color='tab:purple', lw=2, label="Combined LF")
plt.xlim(0, 60)
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.ylabel('Probability density [-]')
plt.xlabel('Delayed time [s]')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# plt.figure(figsize=(20,8))
# plt.title('Combined KDE')
# plt.plot(t_kde_common, hf_kde[hf_imax, :], color='tab:blue', lw=2, label='HF')
# plt.plot(t_kde_common, lf_kde[lf_imax, :], color='tab:orange', lw=2, label='LF')
# plt.xlabel('Delayed time [s]')
# plt.grid(linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()

# +
# replacing kde by histogram
hist_range = (t_kde[0], t_kde[-1])
bins = len(t_kde)
print(t_kde[2]-t_kde[1])
overlap = dt_kde
kernel_bins = int(np.round(bin_width / overlap))
print(kernel_bins)
if kernel_bins % 2 == 0:
    kernel_bins += 1  # Ensure odd length
kernel = np.ones(kernel_bins) / kernel_bins

hist, bin_edges = np.histogram(n_delayed_picks_hf[nhf_imax, :], bins=bins, range=hist_range, weights=nSNRhf)
hist = sp.convolve(hist, kernel, mode="same")
hist = hist / np.trapezoid(hist, t_kde)

s_hist, s_bin_edges = np.histogram(s_delayed_picks_hf[shf_imax, :], bins=bins, range=hist_range, weights=sSNRhf)
s_hist = sp.convolve(s_hist, kernel, mode="same")
s_hist = s_hist / np.trapezoid(s_hist, t_kde)

lf_hist, lf_bin_edges = np.histogram(n_delayed_picks_lf[nlf_imax, :], bins=bins, range=hist_range, weights=nSNRlf)
lf_hist = sp.convolve(lf_hist, kernel, mode="same")
lf_hist = lf_hist / np.trapezoid(lf_hist, t_kde)
s_lf_hist, s_lf_bin_edges = np.histogram(s_delayed_picks_lf[slf_imax, :], bins=bins, range=hist_range, weights=sSNRlf)
s_lf_hist = sp.convolve(s_lf_hist, kernel, mode="same")
s_lf_hist = s_lf_hist / np.trapezoid(s_lf_hist, t_kde)

# +
# Print the delayed time for the maximum KDE on north and south cables, hf 
plt.figure(figsize=(20,8))
plt.subplot(3, 2, (1, 3))
plt.title('North Cable - HF')
plt.scatter(n_delayed_picks_hf[nhf_imax, :], (n_longi_offset + npeakshf[0][:]) * dx * 1e-3, label='LF', c=nSNRhf, s=nSNRhf*0.8, cmap='plasma', rasterized=True)
plt.xlim(min(n_delayed_picks_hf[shf_imax, :]), max(n_delayed_picks_hf[shf_imax, :]))
# plt.xlim(4, 8)
plt.grid(linestyle='--', alpha=0.5)
plt.ylabel('Distance [km]')

plt.subplot(3, 2, (2, 4))
plt.title('South Cable - HF')
plt.scatter(s_delayed_picks_hf[shf_imax, :], (s_longi_offset + speakshf[0][:]) * dx * 1e-3, label='LF', c=sSNRhf, s=sSNRhf*0.8, cmap='plasma', rasterized=True)
plt.xlim(min(s_delayed_picks_hf[shf_imax, :]), max(s_delayed_picks_hf[shf_imax, :]))
plt.grid(linestyle='--', alpha=0.5)
# plt.xlim(4, 8)

plt.subplot(3, 2, 5)
plt.plot(t_kde, n_kde_hf[nhf_imax, :], color='tab:orange', lw=2, label='north HF')
plt.bar(bin_edges[:-1], hist, width=bin_width, alpha=0.5, label="Histogram", color='grey', edgecolor='black')

# plt.plot(t_kde, n_kde_lf[nlf_imax, :], color='tab:orange', lw=2, label='north LF')
# plt.plot(t_kde, n_kde_hf[hf_imax, :], color='tab:cyan', lw=2, ls='--', label='north HF, maxsingle')
# plt.plot(t_kde, n_kde_lf[lf_imax, :], color='tab:red', lw=2, ls='--', label='north LF, maxsingle')
plt.xlim(min(n_delayed_picks_hf[shf_imax, :]), max(n_delayed_picks_hf[shf_imax, :]))
# plt.xlim(4, 8)
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.grid(linestyle='--', alpha=0.5)
plt.ylabel('Picks occurence [-]')
plt.xlabel('Delayed time [s]')

plt.subplot(3, 2, 6)
plt.plot(t_kde, s_kde_hf[shf_imax, :], color='tab:orange', lw=2, label='south HF')
plt.bar(s_bin_edges[:-1], s_hist, width=bin_width, alpha=0.5, label="Histogram", color='grey', edgecolor='black')
plt.xlim(min(s_delayed_picks_hf[shf_imax, :]), max(s_delayed_picks_hf[shf_imax, :]))
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.xlabel('Delayed time [s]')
plt.tight_layout()
plt.grid(linestyle='--', alpha=0.5)
plt.savefig('../figs/delayed_picks_HF.pdf', bbox_inches='tight', transparent=True, format='pdf')
plt.show()
# -

# Print the delayed time for the maximum KDE on north and south cables, lf
plt.figure(figsize=(20,8))
plt.subplot(3, 2, (1, 3))
plt.title('North Cable - LF')
plt.scatter(n_delayed_picks_lf[nlf_imax, :], (n_longi_offset + npeakslf[0][:]) * dx * 1e-3, label='LF', c=nSNRlf, s=nSNRlf*0.8, cmap='plasma', rasterized=True)
plt.xlim(min(n_delayed_picks_lf[nlf_imax, :]), max(n_delayed_picks_lf[nlf_imax, :]))
plt.grid(linestyle='--', alpha=0.5)
plt.ylabel('Distance [km]')
plt.subplot(3, 2, (2, 4))
plt.title('South Cable - LF')       
plt.scatter(s_delayed_picks_lf[slf_imax, :], (s_longi_offset + speakslf[0][:]) * dx * 1e-3, label='LF', c=sSNRlf, s=sSNRlf*0.8, cmap='plasma', rasterized=True)
plt.xlim(min(s_delayed_picks_lf[slf_imax, :]), max(s_delayed_picks_lf[slf_imax, :]))
plt.grid(linestyle='--', alpha=0.5)
plt.subplot(3, 2, 5)        
plt.plot(t_kde, n_kde_lf[nlf_imax, :], color='tab:orange', lw=2, label='north LF')
plt.bar(lf_bin_edges[:-1], lf_hist, width=bin_width, alpha=0.5, label="Histogram", color='grey', edgecolor='black')
# plt.plot(t_kde, n_kde_lf[nlf_imax, :], color='tab:orange', lw=2, label='north LF')
# plt.plot(t_kde, n_kde_hf[hf_imax, :], color='tab:cyan', lw=2, ls='--', label='north HF, maxsingle')
# plt.plot(t_kde, n_kde_lf[lf_imax, :], color='tab:red', lw=2, ls='--', label='north LF, maxsingle')
plt.xlim(min(n_delayed_picks_lf[nlf_imax, :]), max(n_delayed_picks_lf[nlf_imax, :]))        
# plt.xlim(4, 8)
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.grid(linestyle='--', alpha=0.5)
plt.ylabel('Picks occurence [-]')
plt.xlabel('Delayed time [s]')
plt.subplot(3, 2, 6)
plt.plot(t_kde, s_kde_lf[slf_imax, :], color='tab:orange', lw=2, label='south LF')
plt.bar(s_lf_bin_edges[:-1], s_lf_hist, width=bin_width, alpha=0.5, label="Histogram", color='grey', edgecolor='black')
plt.xlim(min(s_delayed_picks_lf[slf_imax, :]), max(s_delayed_picks_lf[slf_imax, :]))
plt.ylim(0, max(np.max(hf_kde), np.max(lf_kde)) * 1.1)
plt.xlabel('Delayed time [s]')
plt.tight_layout()
plt.grid(linestyle='--', alpha=0.5)
plt.savefig('../figs/delayed_picks_LF.pdf', bbox_inches='tight', transparent=True, format='pdf')
plt.show()

max_time_hf = t_kde[hf_tmax]
max_time_lf = t_kde[lf_tmax]

# +
# Plot the hyberbola on top of the picks 
# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False, constrained_layout=True)

# First subplot
sc1 = axes[0, 0].scatter(npeakshf[1][:] / fs, (n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3, 
                         c='grey',  s=nSNRhf, rasterized=True, alpha=0.8)
axes[0, 0].plot(max_time_hf + n_arr_tg[hf_imax, :], n_dist/1e3, ls='-', lw=3, color='tab:blue')
axes[0, 0].plot(max_time_hf + n_arr_tg[hf_imax, :] + dt_sel, n_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[0, 0].plot(max_time_hf + n_arr_tg[hf_imax, :] - dt_sel, n_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[0, 0].set_title('North Cable - HF')
axes[0, 0].set_ylabel('Distance [km]')
axes[0, 0].grid(linestyle='--', alpha=0.5)
axes[0, 0].set_ylim(min(n_dist/1e3), max(n_dist/1e3))

# Second subplot
sc2 = axes[0, 1].scatter(npeakslf[1][:] / fs, (n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3, 
                         c='grey',  s=nSNRlf, rasterized=True, alpha=0.8)
axes[0, 1].plot(max_time_lf + n_arr_tg[lf_imax, :], n_dist/1e3, ls='-', lw=3, color='tab:blue')
axes[0, 1].plot(max_time_lf + n_arr_tg[lf_imax, :] + dt_sel, n_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[0, 1].plot(max_time_lf + n_arr_tg[lf_imax, :] - dt_sel, n_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[0, 1].set_title('North Cable - LF')
axes[0, 1].grid(linestyle='--', alpha=0.5)
axes[0, 1].set_yticklabels([])

# Third subplot
sc3 = axes[1, 0].scatter(speakshf[1][:] / fs, (s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3, 
                         c='grey',  s=sSNRhf, rasterized=True, alpha=0.8)
axes[1, 0].plot(max_time_hf + s_arr_tg[hf_imax, :], s_dist/1e3, ls='-', lw=3, color='tab:blue')
axes[1, 0].plot(max_time_hf + s_arr_tg[hf_imax, :] + dt_sel, s_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[1, 0].plot(max_time_hf + s_arr_tg[hf_imax, :] - dt_sel, s_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[1, 0].set_title('South Cable - HF')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Distance [km]')
axes[1, 0].grid(linestyle='--', alpha=0.5)
# set xlim to the same as the first subplot
axes[1, 0].set_xlim(min(npeakshf[1][:] / fs), max(npeakshf[1][:] / fs))
axes[1, 0].set_ylim(min(s_dist/1e3), max(s_dist/1e3))
axes[1, 1].set_xticks(np.arange(0, max(speakshf[1][:] / fs)+10, 10))


# Fourth subplot
sc4 = axes[1, 1].scatter(speakslf[1][:] / fs, (s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3, 
                         c='grey',  s=sSNRlf, rasterized=True, alpha=0.8)
axes[1, 1].plot(max_time_lf + s_arr_tg[lf_imax, :], s_dist/1e3, ls='-', lw=3, color='tab:blue')
axes[1, 1].plot(max_time_lf + s_arr_tg[lf_imax, :] + dt_sel, s_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[1, 1].plot(max_time_lf + s_arr_tg[lf_imax, :] - dt_sel, s_dist/1e3, ls='--', lw=3, color='tab:orange')
axes[1, 1].set_title('South Cable - LF')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].grid(linestyle='--', alpha=0.5)
# set xlim to the same as the first subplot
axes[1, 1].set_xlim(min(npeakslf[1][:] / fs), max(npeakslf[1][:] / fs))
axes[1, 1].set_yticklabels([])
axes[1, 1].set_xticks(np.arange(0, max(speakslf[1][:] / fs)+10, 10))
plt.savefig('../figs/associated_calls_1st.pdf', bbox_inches='tight', transparent=True, format='pdf')

plt.show()


# -

def associate_picks(kde, t_grid, longi_offset, up_peaks, arr_tg, dx, c0, w_eval, dt_sel, fs, cable_pos, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas, snr):
    """Associates picks with hyperbolas and updates the picks list."""
    # Find the maximum of the KDE
    max_kde_idx = np.argmax(kde)
    imax, tmax = np.unravel_index(max_kde_idx, kde.shape)
    max_time = t_grid[tmax].item()
    # Select the picks that are within the 1.4 s window of the hyperbola
    hyperbola = max_time + arr_tg[imax, :] # Theoretical arrival times for the selected hyperbola
    idx_dist, idx_time = compute_selected_picks(up_peaks, hyperbola, dt_sel, fs) # Select the picks around the hyperbola within +/- dt_sel

    times = idx_time / fs
    distances = (longi_offset + idx_dist) * dx * 1e-3

    window_mask = (times > np.min(times)) & (times < np.min(times) + w_eval)
    # w_times = times[window_mask]
    # w_distances = distances[window_mask]

    # Calulate least squares fit
    idxmin_t = np.argmin(idx_time)
    apex_loc = cable_pos[:, 0][idx_dist[idxmin_t]]
    Ti = idx_time / fs
    Nbiter = 20
    # Initial guess (apex_loc, mean_y, -30m, min(Ti))
    n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

    # Solve the least squares problem
    n, residuals = dw.loc.solve_lq(Ti, cable_pos[idx_dist], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
    # rms residual
    rms = np.sqrt(np.mean(residuals[window_mask]**2))
    
    if rms < .4:
        # Compute the residual cumsum from the minimum time, in positive and negative directions
        #TODO: change variable names
        left_cs = np.cumsum(abs(residuals[idxmin_t::-1])) # negative direction
        right_cs = np.cumsum(abs(residuals[idxmin_t:])) # positive direction
        mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))

        mask_resi = mod_cs < 1500 # Mask the residuals that are below the threshold, key parameter

        associated_list.append(np.asarray((idx_dist[mask_resi], idx_time[mask_resi])))
        used_hyperbolas.append(arr_tg[imax, :])
        arr_tg[imax, :] = dw.loc.calc_arrival_times(0, cable_pos, n[:3], c0)

        # Remove selected picks from updated picks
        # Create a boolean mask that starts by marking every column as True (to keep)
        mask = np.ones(up_peaks.shape[1], dtype=bool)
        for d, t in zip(idx_dist[mask_resi], idx_time[mask_resi]):   # For each pair to remove, update the mask
            mask &= ~((up_peaks[0, :] == d) & (up_peaks[1, :] == t))
        # Apply the mask only once to filter the columns
        up_peaks = up_peaks[:, mask]
        # Remove corresponding snr values
        snr = snr[mask]

    # if compute_curvature(w_times, w_distances) < 1000:
    #     associated_list.append(np.asarray((sidx_dist, sidx_time)))
    #     used_hyperbolas.append(arr_tg[imax, :])

    else:
        # Add the rejected hyperbola to the list
        rejected_list.append(np.asarray((idx_dist, idx_time)))
        rejected_hyperbolas.append(arr_tg[imax, :])
        # Remove the hyperbola from the list
        arr_tg = np.delete(arr_tg, imax, axis=0)

    return up_peaks, arr_tg, associated_list, used_hyperbolas, rejected_list, rejected_hyperbolas, snr


# +

pbar = tqdm(range(iterations), desc="Associated calls: 0")

# Start the loop that runs for a fixed number of iterations
for iteration in pbar:
    # Precompute the time indices
    n_idx_times = np.array(n_up_peaks[1]) / fs # Update with the remaining peaks
    s_idx_times = np.array(s_up_peaks[1]) / fs # Update with the remaining peaks

    # Make a delayed picks array for all the grid points
    # Broadcast the time indices delayed by the theoretical arrival times for the grid points
    n_delayed_picks_hf = n_idx_times[None, :] - n_arr_tg[:, n_up_peaks[0]]
    s_delayed_picks_hf = s_idx_times[None, :] - s_arr_tg[:, s_up_peaks[0]]

    # Generate a 
    global_min = min(np.min(n_delayed_picks_hf), np.min(n_delayed_picks_lf), np.min(s_delayed_picks_hf), np.min(s_delayed_picks_lf))
    global_max = max(np.max(n_delayed_picks_hf), np.max(n_delayed_picks_lf), np.max(s_delayed_picks_hf), np.max(s_delayed_picks_lf))
    Nkde = np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
    t_kde = np.linspace(global_min, global_max, Nkde)

    # Parallelized KDE computation
    n_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(compute_kde)(n_delayed_picks_hf[i, :], t_kde, bin_width) 
        for i in range(n_shape_x)
    ))

    s_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(compute_kde)(s_delayed_picks_hf[i, :], t_kde, bin_width)
        for i in range(s_shape_x)
    ))


    n_up_peaks, n_arr_tg, n_associated_list, n_used_hyperbolas, n_rejected_list, n_rejected_hyperbolas, n_SNR = associate_picks(n_kde_hf, t_kde, n_longi_offset, n_up_peaks, n_arr_tg, dx, c0, w_eval, dt_sel, fs, n_cable_pos, n_associated_list, n_used_hyperbolas, n_rejected_list, n_rejected_hyperbolas, n_SNR)
    n_shape_x = n_arr_tg.shape[0] 
    
    s_up_peaks, s_arr_tg, s_associated_list, s_used_hyperbolas, s_rejected_list, s_rejected_hyperbolas, s_SNR = associate_picks(s_kde_hf, t_kde, s_longi_offset, s_up_peaks, s_arr_tg, dx, c0, w_eval, dt_sel, fs, s_cable_pos, s_associated_list, s_used_hyperbolas, s_rejected_list, s_rejected_hyperbolas, s_SNR)
    s_shape_x = s_arr_tg.shape[0]
    
    pbar.set_description(f"Associated calls: {len(n_associated_list) + len(s_associated_list)}")


print(f"Test completed with {iterations} iterations.")


# +
def plot_reject_pick(peaks, longi_offset, dist, dx, associated_list, rejected_list, rejected_hyperbolas):
    # Plot the selected picks alongside the original picks
    plt.figure(figsize=(20,8))
    plt.subplot(2, 2, 1)
    plt.scatter(peaks[1][:] / fs, (longi_offset + peaks[0][:]) * dx * 1e-3, label='HF', s=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    plt.subplot(2, 2, 2)
    for select in associated_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]') 
    # Plot the deleted hyperbolas
    plt.subplot(2, 2, 3)
    for hyp in rejected_hyperbolas:
        plt.plot(hyp, dist/1e3, label='Rejected hyperbola')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    # plot the rejected picks
    plt.subplot(2, 2, 4)
    for select in rejected_list:
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlabel('Time [s]')
    plt.show()

plot_reject_pick(n_peaks, n_longi_offset, n_dist, dx, n_associated_list, n_rejected_list, n_rejected_hyperbolas)
plot_reject_pick(s_peaks, s_longi_offset, s_dist, dx, s_associated_list, s_rejected_list, s_rejected_hyperbolas)


# +
import numpy as np
import matplotlib.pyplot as plt

def plot_pick_analysis(associated_list, fs, dx, longi_offset, cable_pos, dist, window_size=5, mu_ref=None, sigma_ref=None):
    """
    Create detailed plots of seismic picks with continuity analysis and a normalized curvature score.
    
    Parameters:
    -----------
    associated_list : list
        List of tuples containing pick coordinates and times
    fs : float
        Sampling frequency
    dx : float
        Spatial sampling interval
    longi_offset : float
        Longitudinal offset value
    window_size : float, optional
        Size of analysis window in seconds (default: 5)
    mu_ref : float, optional
        Reference mean curvature for normalization (default: computed from data)
    sigma_ref : float, optional
        Reference standard deviation of curvature for normalization (default: computed from data)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    fig = plt.figure(figsize=(24, 8))
    
    curvature_means = []
    curvature_stds = []
    
    for i, select in enumerate(associated_list):
        times = select[1][:] / fs
        distances = (longi_offset + select[0][:]) * dx * 1e-3
        
        ax = plt.subplot(1, 2*len(associated_list), (i + 1) * 2 - 1)
        if i == 0:
            ax.set_ylabel('Distance [km]')
        ax.scatter(times, distances, label='All Picks', s=0.5, color='gray', alpha=0.5)
        
        window_mask = (times > np.min(times)) & (times < np.min(times) + window_size)
        window_times = times[window_mask]
        window_distances = distances[window_mask]
        
        ax.plot(window_times, window_distances, 
                label='Windowed Picks', 
                lw=2, 
                color='tab:red', 
                alpha=0.6)
        # Calulate least squares fit
        idxmin_t = np.argmin(select[1][:])
        apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
        Ti = select[1][:] / fs
        Nbiter = 20

        # Initial guess (apex_loc, mean_y, -30m, min(Ti))
        n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

        # Solve the least squares problem
        n, residuals = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
        loc_hyerbola = dw.loc.calc_arrival_times(n[-1], cable_pos, n[:3], c0)
        test = np.cumsum(abs(residuals))
        # rms residual
        rms = np.sqrt(np.mean(residuals[window_mask]**2))
        # rms *= 1e4

        left_cs = np.cumsum(abs(residuals[idxmin_t::-1]))
        right_cs = np.cumsum(abs(residuals[idxmin_t:]))
        mod_cs = np.concatenate((left_cs[::-1], right_cs[1:]))

        mask_resi = mod_cs < 1500
        # plot indexes for which only the cumulative sum is less than 1000
        ax.scatter(select[1][mask_resi] / fs, (longi_offset + select[0][mask_resi]) * dx * 1e-3, label='HF', s=1, color='tab:blue')
        ax.plot(loc_hyerbola, dist/1e3, label='Hyperbola', color='tab:green', alpha=0.5)

        # Plot residuals
        # ax.plot(abs(residuals), distances, label='Residuals', color='tab:orange', alpha=0.5)
        # ax.plot(abs(residuals[window_mask]), window_distances, label='Windowed Residuals', color='tab:blue', alpha=0.5)
        # ax.plot(np.cumsum(residuals), distances, label='Cumulative Residuals', color='tab:green', alpha=0.5)
        

        # Calculate curvature
        ddx = np.diff(window_times)
        ddy = np.diff(window_distances)
        ddx2 = np.diff(ddx)
        ddy2 = np.diff(ddy)
        curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
        # curvature = curvature[curvature > 10e-10]
        curvature_mean = np.mean(curvature)

        ax.set_title(f"Pick Analysis\n"
                        f"$\\mu_k$ = {compute_curvature(window_times, window_distances):.2f}\n"
                        f"$\\mu_r$ = {np.mean(abs(residuals[window_mask])):.2f}\n"
                        f"$RMS$ = {rms:.2f}\n",
                        fontsize=10)
        ax.set_xlabel('Time [s]')
        
        ax = plt.subplot(1, 2*len(associated_list), (i + 2) * 2 - 2)
        ax.plot(mod_cs, distances, label='Modified Cumulative Residuals', color='tab:purple', alpha=0.5)
        ax.set_xlabel('Cumulative Residuals')
        
    plt.tight_layout()
    return fig

# Example usage:
fig = plot_pick_analysis(n_associated_list[:10], fs, dx, n_longi_offset, n_cable_pos, n_dist)
fig = plot_pick_analysis(s_associated_list[:10], fs, dx, s_longi_offset, s_cable_pos, s_dist)
plt.show()


# +
# Localize using the selected picks


def loc_from_picks(associated_list, cable_pos, c0, fs):
    localizations = []
    alt_localizations = []

    for select in associated_list:
        idxmin_t = np.argmin(select[1][:])
        apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
        Ti = select[1][:] / fs
        Nbiter = 20

        # Initial guess (apex_loc, mean_y, -30m, min(Ti))
        n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]
        print(f'Initial guess: {n_init[0]:.2f} m, {n_init[1]:.2f} m, {n_init[2]:.2f} m, {n_init[3]:.2f} s')
        # Solve the least squares problem
        n = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter, fix_z=True, ninit=n_init)
        nalt = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter-1, fix_z=True, ninit=n_init)

        localizations.append(n)
        alt_localizations.append(nalt)

    return localizations, alt_localizations

n_localizations, n_alt_localizations = loc_from_picks(n_associated_list, n_cable_pos, c0, fs)
s_localizations, s_alt_localizations = loc_from_picks(s_associated_list, s_cable_pos, c0, fs)


# +
def plot_associated(peaks, longi_offset, associated_list, localizations, cable_pos, dist, dx, c0):
    plt.figure(figsize=(20,8))

    # Plot the time picks with colored associated ones
    plt.subplot(1, 2, 1)
    plt.scatter(peaks[1][:] / fs, (longi_offset + peaks[0][:]) * dx * 1e-3, label='LF', s=0.5, alpha=0.2, color='tab:grey')
    for i, select in enumerate(associated_list):
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
    plt.xlim(0, 60)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')

    # Plot the time picks with the the predicted hyperbola
    plt.subplot(1, 2, 2)
    for i, select in enumerate(associated_list):
        plt.scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3, label='LF', s=0.5)
        plt.plot(dw.loc.calc_arrival_times(localizations[i][-1], cable_pos, localizations[i][:3], c0), dist/1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        # plt.plot(select[1][:] / fs, dw.loc.calc_arrival_times(0, cable_pos, alt_localizations[i][:3], c0), color='tab:orange', ls='-', lw=1)
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    plt.show()

plot_associated(n_peaks, n_longi_offset, n_associated_list, n_localizations, n_cable_pos, n_dist, dx, c0)
plot_associated(s_peaks, s_longi_offset, s_associated_list, s_localizations, s_cable_pos, s_dist, dx, c0)

# +
# Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
opticald_n = []
opticald_s = []

disp_step = 10000 # [m]
dx_ch = n_ds.attrs['dx'] # [m]
idx_step = int(disp_step / dx_ch)

for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
    opticald_n.append((df_north['x'][i], df_north['y'][i]))

for i in range(int(idx_step-df_south["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
    opticald_s.append((df_south['x'][i], df_south['y'][i]))
    
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

# Plot the location of the apex
plt.figure(figsize=(14, 7))
ax = plt.gca()
# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable')
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable')
# ax.plot(cable_pos[j_hf_call[i]][:,0], cable_pos[j_hf_call[i]][:,1], 'tab:green', label='used_cable')

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


for i, loc in enumerate(n_localizations):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], 'o',  c='tab:purple', lw=4, label='Localized call - north')
    else:
        ax.plot(loc[0], loc[1], 'o', c='tab:purple', lw=4)
for i, loc in enumerate(s_localizations):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], 'o', c='tab:green', label='Localized call - south', lw=4)
    else:
        ax.plot(loc[0], loc[1], 'o', c='tab:green', lw=4)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
# Calculate width of image over height
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# plt.xlim(40000, 34000)
# plt.ylim(15000, 25000)
plt.legend(loc='upper left')
plt.grid(linestyle='--', alpha=0.6, color='k')
plt.tight_layout()
plt.show()
