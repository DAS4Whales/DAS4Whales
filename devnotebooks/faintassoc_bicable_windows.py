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

# # Bicable association process with spatial windows for faint calls

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("reload_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# +
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
import colorcet as cc

plt.rcParams['font.size'] = 14

# +
# Load the peak indexes and the metadata

# Well-behaving data 
# n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
# s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

# Gabor filtered data
n_ds = xr.load_dataset('../out/sparse_picks_Gabor/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc')
s_ds = xr.load_dataset('../out/sparse_picks_Gabor/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

# Problematic data
# n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_08:00:02_ipi3_th_4.nc') 
# s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_08:00:02_ipi3_th_5.nc')

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
snr_thresh = 4

speakshf = speakshf[:, sSNRhf > snr_thresh]
speakslf = speakslf[:, sSNRlf > snr_thresh]

sSNRhf = sSNRhf[sSNRhf > snr_thresh]
sSNRlf = sSNRlf[sSNRlf > snr_thresh]
peaks = (npeakshf, npeakslf, speakshf, speakslf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)

dw.assoc.plot_peaks(peaks, SNRs, selected_channels_m, dx, fs)
plt.show()

# +
# Sort the peaks based on SNR difference
npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
)

speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
)

# -

plt.rcParams['font.size'] = 24
# Plot the sorted peaks
peaks = (npeakshf, npeakslf, speakshf, speakslf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)
# dw.assoc.plot_peaks(peaks, SNRs, selected_channels_m, dx, fs)
fig=dw.assoc.plot_tpicks_resolved(peaks, SNRs, selected_channels_m, dx, fs)
plt.savefig('../figs/Figure2.pdf', bbox_inches='tight', transparent=True)
plt.show()


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

# +
# dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)

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
zg = -30

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
# Plot the distance of the grid points to the cable
n_distances = np.sqrt((xg[:, None] - n_cable_pos[:, 0])**2 + (yg[:, None] - n_cable_pos[:, 1])**2 + (zg - n_cable_pos[:, 2])**2)
s_distances = np.sqrt((xg[:, None] - s_cable_pos[:, 0])**2 + (yg[:, None] - s_cable_pos[:, 1])**2 + (zg - s_cable_pos[:, 2])**2)

distances = n_distances.min(axis=1) + s_distances.min(axis=1)
distances *= 0.5 # Average distances to both cables

from matplotlib.patches import Rectangle

# # Plot the distances
# plt.figure(figsize=(24, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(xg, yg, c=distances, cmap='viridis', s=10, edgecolor='none')
# plt.plot(df_north['x'], df_north['y'], color='orange', label='North cable')
# plt.plot(df_south['x'], df_south['y'], color='red', label='South cable')
# # Reverse x-axis to have the origin at the bottom left
# plt.gca().invert_xaxis()
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.xlim(np.max(xg), np.min(xg))
# plt.ylim(np.min(yg), np.max(yg))
# # plt.axis('equal')       
# plt.show()

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(n_dist/1e3, n_cable_pos[:, 2], color='orange', lw=3, label='North cable')
ax.plot(s_dist/1e3, s_cable_pos[:, 2], color='red', lw=3, label='South cable')
# Plot overlapping spatial windows 

ax.add_patch(Rectangle((np.min(n_dist/1e3), np.min(s_cable_pos[:, 2])), 56000/1e3 - np.min(n_dist/1e3), -np.min(s_cable_pos[:, 2]),
             edgecolor = 'black',
             facecolor='tab:grey',
             fill=True,
             alpha=0.3,
             lw=5))

ax.add_patch(Rectangle((35000/1e3, np.min(s_cable_pos[:, 2])), np.max(n_dist/1e3) - 35000/1e3, -np.min(s_cable_pos[:, 2]),
             edgecolor = 'tab:blue',    
             facecolor='tab:blue',
             alpha=0.3,  
             fill=True,
             lw=5))


ax.add_patch(Rectangle((56000/1e3, np.min(s_cable_pos[:, 2])), np.max(s_dist/1e3) - 56000/1e3, -np.min(s_cable_pos[:, 2]),
             edgecolor = 'tab:green',
                facecolor='tab:green',
                alpha=0.3,
             fill=True,
             lw=5))

plt.xlabel('Distance along the cable (km)')
plt.ylabel('Depth (m)')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.ylim(-800, 0)
fig.savefig('../figs/Figure7a.pdf', bbox_inches='tight', transparent=True)

plt.show()

# +
import cmocean.cm as cmo

def plot_cables2D_m_rectPatches(df_north, df_south, bathy, xm, ym):
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

    disp_step = 10000 # [m]
    dx_ch = 2.0419 # [m]
    idx_step = int(disp_step / dx_ch)

    for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
        opticald_n.append((df_north['x'][i], df_north['y'][i]))

    for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
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

    fig = plt.figure(figsize=(14, 9))
    ax = plt.gca()
    # Plot the bathymetry relief in background
    rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
    plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)

    ax.plot(df_north['x'] , df_north['y'] , 'tab:red', label='North cable', lw=2.5)
    ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

    # Add dashed contours at selected depths with annotations
    # depth_levels = [-1500, -1000, -600, -250, -80]

    # contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
    # ax.clabel(contour_dashed, fmt='%d m', inline=True)

    # Plot points along the cable every 10 km in terms of optical distance
    for i, point in enumerate(opticald_n, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 10), ha='center')

    for i, point in enumerate(opticald_s, start=1):
        ax.plot(point[0], point[1], '.', color='k')
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -30), ha='center')

    # Plot the three rectangles representing the spatial windows
    ax.add_patch(Rectangle((np.min(df_north['x']), np.min(df_south['y'])), 
                           56000 - np.min(df_north['x']), 
                           np.max(df_north['y'])-np.min(df_south['y']),
                           edgecolor='black',
                           facecolor='tab:grey',
                           fill=True,
                           alpha=0.3,
                           lw=5))

    ax.add_patch(Rectangle((35000, np.min(df_south['y'])),
                           np.max(df_north['x']) - 35000, 
                           np.max(df_north['y'])-np.min(df_south['y']),
                           edgecolor='tab:blue',    
                           facecolor='tab:blue',
                           alpha=0.3,  
                           fill=True,
                           lw=5))

    ax.add_patch(Rectangle((56000, np.min(df_south['y'])),
                           np.max(df_south['x']) - 56000, 
                           np.max(df_north['y'])-np.min(df_south['y']),
                           edgecolor='tab:green',
                           facecolor='tab:green',
                           alpha=0.3,
                           fill=True,
                           lw=5))

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
    # Get current tick locations and convert to km
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    # Set new tick labels in km
    ax.set_xticklabels([f'{int(tick/1000)}' for tick in x_ticks])
    ax.set_yticklabels([f'{int(tick/1000)}' for tick in y_ticks])

    # Update axis labels
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    return fig

fig2 = plot_cables2D_m_rectPatches(df_north, df_south, bathy, x, y)
fig2.savefig('../figs/Figure7b.pdf', bbox_inches='tight', transparent=True)

# +
# apply the spatial windows to the peaks

print(np.min(n_dist), np.max(n_dist))
win_close = [np.min(n_dist), 56000]
win_mid = [35000, np.max(n_dist)]
win_far = [56000, np.max(s_dist)]

# Convert windows to indexes
win_close = [int(win_close[0] / dx)-n_longi_offset, int(win_close[1] / dx)-n_longi_offset]
win_mid = [int(win_mid[0] / dx)-n_longi_offset, int(win_mid[1] / dx)-n_longi_offset]
win_far = [int(win_far[0] / dx)-n_longi_offset, int(win_far[1] / dx)-n_longi_offset]

def apply_spatial_windows(peaks, snr, win):
    """
    Apply the spatial windows to the peaks.

    Parameters
    ----------
    peaks : tuple of np.ndarray
        The peaks indexes for the North and South cables.
    win : list of float
        The spatial window to apply.

    Returns
    -------
    tuple of np.ndarray
        The peaks indexes after applying the spatial window.
    """
    
    npeakshf, npeakslf, speakshf, speakslf = peaks
    nSNRhf, nSNRlf, sSNRhf, sSNRlf = snr
    
    # Apply the spatial window to the North cable peaks
    mask_hf = (npeakshf[0, :] >= win[0]) & (npeakshf[0, :] <= win[1])
    mask_lf = (npeakslf[0, :] >= win[0]) & (npeakslf[0, :] <= win[1])

    # Filter columns (preserve 2D structure)
    npeakshf = npeakshf[:, mask_hf]
    nSNRhf = nSNRhf[mask_hf]
    npeakslf = npeakslf[:, mask_lf]
    nSNRlf = nSNRlf[mask_lf]

    # Apply the spatial window to the South cable peaks
    mask_hf = (speakshf[0, :] >= win[0]) & (speakshf[0, :] <= win[1])
    mask_lf = (speakslf[0, :] >= win[0]) & (speakslf[0, :] <= win[1])

    speakshf = speakshf[:, mask_hf]
    sSNRhf = sSNRhf[mask_hf]
    speakslf = speakslf[:, mask_lf]
    sSNRlf = sSNRlf[mask_lf]

    peaks = (npeakshf, npeakslf, speakshf, speakslf)
    snr = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
    return peaks, snr

# # Apply the spatial windows to the peaks
peaks_close, snr_close = apply_spatial_windows(peaks, SNRs, win_close)
peaks_mid, snr_mid = apply_spatial_windows(peaks, SNRs, win_mid)
peaks_far, snr_far = apply_spatial_windows(peaks, SNRs, win_far)

# # Plot the peaks after applying the spatial windows
# dw.assoc.plot_peaks(peaks_close, snr_close, selected_channels_m, dx, fs)

# dw.assoc.plot_peaks(peaks_mid, snr_mid, selected_channels_m, dx, fs)

# dw.assoc.plot_peaks(peaks_far, snr_far, selected_channels_m, dx, fs)

# plt.show()
# -


dt_kde = 0.5 # [s] Time resolution of the KDE
bin_width = 1
# dt_kde = 0.25 # [s] Time resolution of the KDE (overlap)
# bin_width = 1.5
dt_tol = int(0.5 * fs) # [samples] Tolerance for the time index when removing picks
# dist_tol = int(10/dx)
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
rms_threshold = 0.5
# Set the number of iterations for testing
iterations = 50

# +
n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)

n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# +
nhf_assoc_list_pair = [] # List to store paired associated picks for the North cable, HF calls
nlf_assoc_list_pair = [] # List to store paired associated picks for the North cable, LF calls
nhf_assoc_list = [] # List to store associated picks for the North cable, HF calls
nlf_assoc_list = [] # List to store associated picks for the North cable, LF calls
n_used_hyperbolas = []
n_rejected_list = []
n_rejected_hyperbolas = []

shf_assoc_list_pair = [] # List to store paired associated picks for the South cable, HF calls
slf_assoc_list_pair = [] # List to store paired associated picks for the South cable, LF calls
shf_assoc_list = [] # List to store associated picks for the South cable, HF calls
slf_assoc_list = [] # List to store associated picks for the South cable, LF calls
s_used_hyperbolas = []
s_rejected_list = []
s_rejected_hyperbolas = []

association_lists = [
    nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair,
    nhf_assoc_list, shf_assoc_list, nlf_assoc_list, slf_assoc_list
    ]

hyperbolas = [n_used_hyperbolas, s_used_hyperbolas]

rejected_lists = [
    n_rejected_list, s_rejected_list, n_rejected_hyperbolas, s_rejected_hyperbolas
]

# +
pbar = tqdm(range(iterations), desc="Associated calls: 0")

for iteration in pbar:
    results = dw.assoc.process_iteration(
    # Peak data
    n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    # Grid data
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x,
    # Cable positions
    n_cable_pos, s_cable_pos, n_longi_offset, s_longi_offset,
    # Association lists
    association_lists,
    # Hyperbolas
    hyperbolas,
    # Rejected lists
    rejected_lists,
    # Parameters
    fs, dt_kde, bin_width, dt_sel, w_eval, rms_threshold, c0, dx, dt_tol,
    # Iteration info
    iteration)

    if results is None:
        print(f"Stopped association at iteration {iteration}.")
        break  # Skip to the next iteration if no results are returned

    (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x, 
    association_lists, rejected_lists, hyperbolas) = results

    total_associations = sum(len(lst) for lst in association_lists)
    pbar.set_description(f"Associated calls: {total_associations}")

# +
# apply the spatial windows to the remaining peaks
up_peaks = (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)

print(n_up_peaks_hf.shape, nSNRhf.shape)
print(s_up_peaks_hf.shape, sSNRhf.shape)
print(n_up_peaks_lf.shape, nSNRlf.shape)
print(s_up_peaks_lf.shape, sSNRlf.shape)
# -

peaks_close, snr_close = apply_spatial_windows(up_peaks, SNRs, win_close)
peaks_mid, snr_mid = apply_spatial_windows(up_peaks, SNRs, win_mid)
peaks_far, snr_far = apply_spatial_windows(up_peaks, SNRs, win_far)

dw.assoc.plot_peaks(peaks_far, snr_far, selected_channels_m, dx, fs)
plt.show()

# +
n_up_peaks_hf = np.copy(peaks_far[0])
s_up_peaks_hf = np.copy(peaks_far[2])
n_up_peaks_lf = np.copy(peaks_far[1])
s_up_peaks_lf = np.copy(peaks_far[3])

nSNRhf = np.copy(snr_far[0])
nSNRlf = np.copy(snr_far[1])
sSNRhf = np.copy(snr_far[2])
sSNRlf = np.copy(snr_far[3])
# -

iterations_far = 50
w_eval_far = 2 
rms_threshold_far = 0.25

# +
pbar = tqdm(range(iterations_far), desc="Associated calls, far window: 0")

for iteration in pbar:
    results = dw.assoc.process_iteration(
    # Peak data
    n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    # Grid data
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x,
    # Cable positions
    n_cable_pos, s_cable_pos, n_longi_offset, s_longi_offset,
    # Association lists
    association_lists,
    # Hyperbolas
    hyperbolas,
    # Rejected lists
    rejected_lists,
    # Parameters
    fs, dt_kde, bin_width, dt_sel, w_eval_far, rms_threshold, c0, dx, dt_tol,
    # Iteration info
    iteration)

    if results is None:
        print(f"Stopped association at iteration {iteration}.")
        break  # Exit the loop if no results are returned

    (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x, 
    association_lists, rejected_lists, hyperbolas) = results

    total_associations = sum(len(lst) for lst in association_lists)
    pbar.set_description(f"Associated calls, far window: {total_associations}")

# +
dw.assoc.plot_reject_pick(npeakshf, n_longi_offset, n_dist, dx, nhf_assoc_list_pair, n_rejected_list, n_rejected_hyperbolas, fs)
# dw.assoc.plot_reject_pick(npeakshf, n_longi_offset, n_dist, dx, nhf_assoc_list, n_rejected_list, n_rejected_hyperbolas, fs)

# dw.assoc.plot_reject_pick(npeakslf, n_longi_offset, n_dist, dx, nlf_assoc_list_pair, n_rejected_list, n_rejected_hyperbolas, fs)
dw.assoc.plot_reject_pick(speakshf, s_longi_offset, s_dist, dx, shf_assoc_list_pair, s_rejected_list, s_rejected_hyperbolas, fs)
# dw.assoc.plot_reject_pick(speakslf, s_longi_offset, s_dist, dx, slf_assoc_list_pair, s_rejected_list, s_rejected_hyperbolas, fs)

# dw.assoc.plot_reject_pick(npeakslf, n_longi_offset, n_dist, dx, nlf_assoc_list, n_rejected_list, n_rejected_hyperbolas, fs)
dw.assoc.plot_reject_pick(speakshf, s_longi_offset, s_dist, dx, shf_assoc_list, s_rejected_list, s_rejected_hyperbolas, fs)
# dw.assoc.plot_reject_pick(speakslf, s_longi_offset, s_dist, dx, slf_assoc_list, s_rejected_list, s_rejected_hyperbolas, fs)

# +
import numpy as np
import matplotlib.pyplot as plt

def plot_pick_analysis(associated_list, fs, dx, longi_offset, cable_pos, dist, snr_tot, peaks, window_size=5, mu_ref=None, sigma_ref=None):
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
    for i, select in enumerate(associated_list):
        times = select[1][:] / fs
        distances = (longi_offset + select[0][:]) * dx * 1e-3
        # snr = dw.assoc.select_snr(peaks, select, snr_tot)

        # Determine common color scale
        # vmin = np.min(snr)
        # vmax = np.max(snr)
        # cmap = cm.plasma  # Define colormap
        # norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Normalize color range

        
        ax = plt.subplot(1, 2*len(associated_list), (i + 1) * 2 - 1)
        if i == 0:
            ax.set_ylabel('Distance [km]')
        ax.scatter(times, distances, label='All Picks', 
                    c='gray', alpha=0.5, s=1)
        
        idxmin_t = np.argmin(select[1][:])
        mask_dist = abs(distances - distances[idxmin_t]) < 40 # limit the distance to 40 km from the minimum

        times = times[mask_dist]
        distances = distances[mask_dist]
        
        window_mask = (times > np.min(times)) & (times < np.min(times) + window_size)
        window_times = times[window_mask]
        window_distances = distances[window_mask]
        
        # ax.scatter(window_times, window_distances, 
        #         label='Windowed Picks', 
        #         lw=2, 
        #         color='tab:red', 
        #         alpha=0.6)
        
        # Calulate least squares fit
        apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
        Ti = times
        Nbiter = 20


        # Initial guess (apex_loc, mean_y, -30m, min(Ti))
        n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]

        # Solve the least squares problem
        n, residuals = dw.loc.solve_lq_weight(Ti, cable_pos[select[0][mask_dist]], c0, Nbiter, fix_z=True, ninit=n_init, residuals=True)
        loc_hyerbola = dw.loc.calc_arrival_times(n[-1], cable_pos, n[:3], c0)

        # rms residual
        rms = np.sqrt(np.mean(residuals[window_mask]**2))
        rms_total = np.sqrt(np.mean(residuals**2))
        mask_resi = abs(residuals) < 2 * rms_total
        # gaps_w = np.diff(window_distances)

        gaps = np.zeros_like(distances)
        # Find the gaps only for the valid (masked) distances
        valid_distances = distances[mask_resi]
        if valid_distances.size > 1:
            gaps_valid = np.abs(np.diff(valid_distances))
            # Assign the gaps to the correct positions
            idx_valid = np.flatnonzero(mask_resi)
            gaps[idx_valid[:-1]] = gaps_valid
        print(np.shape(mask_resi), np.shape(gaps), np.shape(distances))

        # Remove points after a large gap from the minimum 
        # Remove points after a large gap from the minimum 
        gap_tresh = 4 # km 
        for l, gap in enumerate(gaps[idxmin_t:]):
            if gap > gap_tresh:
                mask_resi[idxmin_t + l + 1:] = False
                break

        # Remove points before a large gap from the minimum 
        for l, gap in enumerate(gaps[idxmin_t-1::-1]):
            if gap > gap_tresh:
                mask_resi[:idxmin_t - l] = False
                break

        # ax.scatter(select[1][mask_resi] / fs, (longi_offset + select[0][mask_resi]) * dx * 1e-3, label='HF', s=4, color='tab:blue')
        ax.scatter(times[mask_resi], distances[mask_resi], label='HF', s=4, color='tab:blue')
        ax.scatter(select[1][mask_dist] / fs, (longi_offset + select[0][mask_dist]) * dx * 1e-3, label='LF', s=8, color='tab:orange')
        ax.plot(loc_hyerbola, dist/1e3, label='Hyperbola', color='tab:green', alpha=0.5)

        ax.set_title(f"Pick Analysis\n"
                        f"$\\mu_r$ = {np.mean(abs(residuals[window_mask])):.2f}\n"
                        f"$RMS_w$ = {rms:.2f}\n"
                        f"$RMS_t$ = {rms_total:.2f}\n",
                        fontsize=10)
        ax.set_xlabel('Time [s]')
        
        ax = plt.subplot(1, 2*len(associated_list), (i + 2) * 2 - 2)
        # ax.plot(abs(residuals), distances, label='Modified Cumulative Residuals', color='tab:purple', alpha=0.5)
        ax.plot(np.diff(distances), distances[:-1], label='Curvature', color='tab:orange', alpha=0.5)
        ax.plot(gaps, distances, label='Distance Gaps', color='tab:blue', alpha=0.5)
        # ax.plot(np.diff(window_distances), window_distances[:-1], label='Windowed Curvature', color='tab:purple', alpha=0.5)

        ax.set_xlabel('Distance gaps [km]')
        
    plt.tight_layout()
    return fig

# Example usage:
# fig = plot_pick_analysis(nhf_assoc_list, fs, dx, n_longi_offset, n_cable_pos, n_dist)
# fig = plot_pick_analysis(nlf_assoc_list_pair, fs, dx, n_longi_offset, n_cable_pos, n_dist)
# fig = plot_pick_analysis(n_rejected_list, fs, dx, n_longi_offset, n_cable_pos, n_dist)
# fig = plot_pick_analysis(nlf_assoc_list, fs, dx, n_longi_offset, n_cable_pos, n_dist)
# fig = plot_pick_analysis(shf_assoc_list, fs, dx, s_longi_offset, s_cable_pos, s_dist)
# fig = plot_pick_analysis(slf_assoc_list, fs, dx, s_longi_offset, s_cable_pos, s_dist)
# fig = plot_pick_analysis(slf_assoc_list_pair, fs, dx, s_longi_offset, s_cable_pos, s_dist)
sSNRhf = s_ds["SNR_hf"].values
sSNRhf = sSNRhf[sSNRhf > 5]
sSNRlf = s_ds["SNR_lf"].values
sSNRlf = sSNRlf[sSNRlf > 5]
nSNRhf = n_ds["SNR_hf"].values
nSNRlf = n_ds["SNR_lf"].values
# fig = plot_pick_analysis(shf_assoc_list_pair, fs, dx, s_longi_offset, s_cable_pos, s_dist, sSNRhf, speakshf)
# fig = plot_pick_analysis(slf_assoc_list_pair, fs, dx, s_longi_offset, s_cable_pos, s_dist, sSNRlf, speakslf)
# fig = plot_pick_analysis(nhf_assoc_list_pair, fs, dx, n_longi_offset, n_cable_pos, n_dist, nSNRhf, npeakshf)
# fig = plot_pick_analysis(nlf_assoc_list_pair, fs, dx, n_longi_offset, n_cable_pos, n_dist, nSNRlf, npeakslf)
# fig = plot_pick_analysis(nhf_assoc_list, fs, dx, n_longi_offset, n_cable_pos, n_dist, nSNRhf, npeakshf)
plt.show()

# -

fig = plot_pick_analysis(s_rejected_list[:5], fs, dx, s_longi_offset, s_cable_pos, s_dist, sSNRhf, speakshf)
fig = plot_pick_analysis(n_rejected_list[:5], fs, dx, n_longi_offset, n_cable_pos, n_dist, nSNRhf, npeakshf)
plt.show()


# +
# Localize using the selected picks
# nSNRhf = n_ds["SNR_hf"].values
# nSNRlf = n_ds["SNR_lf"].values
# sSNRhf = s_ds["SNR_hf"].values
# sSNRlf = s_ds["SNR_lf"].values

def select_snr(up_peaks, selected_peaks, snr):
    print(np.shape(up_peaks), np.shape(selected_peaks), np.shape(snr))
    # Start with a mask of all True
    mask = np.zeros(up_peaks.shape[1], dtype=bool)

    # Accumulate the mask for each selected pair (d, t)
    for d, t in zip(selected_peaks[0], selected_peaks[1]):
        mask |= (up_peaks[0, :] == d) & (up_peaks[1, :] == t)

    # Return the snr values for the selected (d, t) pairs
    return snr[mask]

# def loc_from_picks_list(associated_list, cable_pos, c0, fs, peaks, snr):
#     localizations = []
#     alt_localizations = []
#     for select in associated_list:
#         idxmin_t = np.argmin(select[1][:])
#         apex_loc = cable_pos[:, 0][select[0][idxmin_t]]
#         Ti = select[1][:] / fs
#         Nbiter = 20
#         # Select SNR values for the current selection
#         snr_i = select_snr(peaks, select, snr)
#         print(np.shape(snr_i))
#         # Initial guess (apex_loc, mean_y, -30m, min(Ti))
#         n_init = [apex_loc, np.mean(cable_pos[:,1]), -40, np.min(Ti)]
#         print(f'Initial guess: {n_init[0]:.2f} m, {n_init[1]:.2f} m, {n_init[2]:.2f} m, {n_init[3]:.2f} s')
#         # Solve the least squares problem
#         n = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter, SNR=snr_i, fix_z=True, ninit=n_init)
#         nalt = dw.loc.solve_lq(Ti, cable_pos[select[0][:]], c0, Nbiter-1, fix_z=True, ninit=n_init)

#         localizations.append(n)
#         alt_localizations.append(nalt)

#     return localizations, alt_localizations

def loc_from_picks_list(associated_list, cable_pos, c0, fs):
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
        n = dw.loc.solve_lq_weight(Ti, cable_pos[select[0][:]], c0, Nbiter, fix_z=True, ninit=n_init)
        nalt = dw.loc.solve_lq_weight(Ti, cable_pos[select[0][:]], c0, Nbiter-1, fix_z=True, ninit=n_init)

        localizations.append(n)
        alt_localizations.append(nalt)

    return localizations, alt_localizations

# nhf_localizations, nhf_alt_localizations = loc_from_picks_list(nhf_assoc_list, n_cable_pos, c0, fs, npeakshf, nSNRhf)
# nlf_localizations, nlf_alt_localizations = loc_from_picks_list(nlf_assoc_list, n_cable_pos, c0, fs, npeakslf, nSNRlf)
# shf_localizations, s_alt_localizations = loc_from_picks_list(shf_assoc_list, s_cable_pos, c0, fs, speakshf, sSNRhf)
# slf_localizations, s_alt_localizations = loc_from_picks_list(slf_assoc_list, s_cable_pos, c0, fs, speakslf, sSNRlf)


# +
dw.assoc.clean_pairs(nhf_assoc_list_pair, shf_assoc_list_pair, shf_assoc_list)
dw.assoc.clean_pairs(nlf_assoc_list_pair, slf_assoc_list_pair, slf_assoc_list)
dw.assoc.clean_pairs(shf_assoc_list_pair, nhf_assoc_list_pair, nhf_assoc_list)
dw.assoc.clean_pairs(slf_assoc_list_pair, nlf_assoc_list_pair, nlf_assoc_list)

dw.assoc.clean_singles(nhf_assoc_list)
dw.assoc.clean_singles(nlf_assoc_list)
dw.assoc.clean_singles(shf_assoc_list)
dw.assoc.clean_singles(slf_assoc_list)


# -

def plot_associated_bicable(n_peaks, s_peaks, longi_offset, pair_assoc_list, pair_loc_list, associated_list, localizations,
                            n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs):
    
    nhf_assoc_pair, nlf_assoc_pair, shf_assoc_pair, slf_assoc_pair = pair_assoc_list
    nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list = associated_list
    nhf_loc_pair, nlf_loc_pair, shf_loc_pair, slf_loc_pair = pair_loc_list
    nhf_localizations, nlf_localizations, shf_localizations, slf_localizations = localizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False, constrained_layout=True)

    # Get color palettes
    hf_palette = plt.get_cmap('YlOrRd_r')
    lf_palette = plt.get_cmap('YlGnBu_r')

    # Assign color per HF/LF event
    nbhf = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list) + len(shf_assoc_list)
    nblf = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list) + len(slf_assoc_list)

    start, end = 0.0, 0.6  # Avoids part of the coolormap that is too light

    hf_colors = [hf_palette(start + (end - start) * i / max(nbhf - 1, 1)) for i in range(nbhf)]
    lf_colors = [lf_palette(start + (end - start) * i / max(nblf - 1, 1)) for i in range(nblf)]

    # First subplot — North raw picks and associated
    # -- Raw picks --
    axes[0, 0].scatter(n_peaks[1][:] / fs, (longi_offset + n_peaks[0][:]) * dx * 1e-3,
                       label='All peaks', s=0.5, alpha=0.2, color='tab:grey', rasterized=True)
    # -- Associated picks - pairs --
    for i, select in enumerate(nhf_assoc_pair):
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(nlf_assoc_pair):
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        
    # -- Associated picks - single --
    for i, select in enumerate(nhf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair)
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
    for i, select in enumerate(nlf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair)
        # print(i, idx_offset, len(nhf_assoc_pair), len(shf_assoc_pair), len(nhf_assoc_list)print(len(lf_colors), len(nlf_assoc_list)))
        axes[0, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
    axes[0, 0].set_title('North')       
    axes[0, 0].set_ylabel('Distance [km]')
    axes[0, 0].set_xlim(0, 70)

    # Second subplot — North with arrival curves
    # -- Associated picks - pairs --
    for i, select in enumerate(nhf_assoc_pair):
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nhf_loc_pair[i][-1], n_cable_pos, 
                                                  nhf_loc_pair[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
                                                  
    for i, select in enumerate(nlf_assoc_pair):
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nlf_loc_pair[i][-1], n_cable_pos,
                                                  nlf_loc_pair[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    # -- Associated picks - single --
    for i, select in enumerate(nhf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair)
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nhf_localizations[i][-1], n_cable_pos,
                                                  nhf_localizations[i][:3], c0),
                                                  n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(nlf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair)
        axes[0, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
        axes[0, 1].plot(dw.loc.calc_arrival_times(nlf_localizations[i][-1], n_cable_pos,
                                                  nlf_localizations[i][:3], c0),
                        n_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
    # Remove the y-axis ticks labels
    axes[0, 1].set_yticklabels([])

    # Third subplot — South raw picks and associated
    # -- Raw picks --
    axes[1, 0].scatter(s_peaks[1][:] / fs, (longi_offset + s_peaks[0][:]) * dx * 1e-3,
                       label='All peaks', s=0.5, alpha=0.2, color='tab:grey', rasterized=True)
    # -- Associated picks - pairs --
    for i, select in enumerate(shf_assoc_pair):
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(slf_assoc_pair):
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        
    # -- Associated picks - single --
    for i, select in enumerate(shf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list)
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        
    for i, select in enumerate(slf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list)
        axes[1, 0].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
    axes[1, 0].set_title('South')
    axes[1, 0].set_ylabel('Distance [km]')
    axes[1, 0].set_xlabel('Time [s]')

    # Fourth subplot — South with arrival curves
    # -- Associated picks - pairs --
    for i, select in enumerate(shf_assoc_pair):
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i], s=10, marker='>', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(shf_loc_pair[i][-1], s_cable_pos,
                                                  shf_loc_pair[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(slf_assoc_pair):
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i], s=10, marker='o', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(slf_loc_pair[i][-1], s_cable_pos,
                                                  slf_loc_pair[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    # -- Associated picks - single --
    for i, select in enumerate(shf_assoc_list):
        idx_offset = len(nhf_assoc_pair) + len(shf_assoc_pair) + len(nhf_assoc_list)
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=hf_colors[i+idx_offset], s=10, marker='>', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(shf_localizations[i][-1], s_cable_pos,
                                                  shf_localizations[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)
        
    for i, select in enumerate(slf_assoc_list):
        idx_offset = len(nlf_assoc_pair) + len(slf_assoc_pair) + len(nlf_assoc_list)
        axes[1, 1].scatter(select[1][:] / fs, (longi_offset + select[0][:]) * dx * 1e-3,
                           color=lf_colors[i+idx_offset], s=10, marker='o', rasterized=True)
        axes[1, 1].plot(dw.loc.calc_arrival_times(slf_localizations[i][-1], s_cable_pos,
                                                  slf_localizations[i][:3], c0),
                        s_dist / 1e3, color='tab:grey', ls='-', lw=2, alpha=0.7)

    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_yticklabels([])

    # Add a common legend
    hf_handle = plt.Line2D([], [], marker='>', color='w', label='HF calls',
                           markerfacecolor='tab:red', markersize=10)
    lf_handle = plt.Line2D([], [], marker='o', color='w', label='LF calls',
                           markerfacecolor='tab:blue', markersize=10)
    

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    # Add gradient legend to one of your subplots
    gradient_values = np.linspace(start, end, 100).reshape(1, -1)
    hf_cmap_custom = ListedColormap(hf_colors)
    lf_cmap_custom = ListedColormap(lf_colors)

    # Create a parent container for the legend with frame
    legend_container = inset_axes(axes[1, 1], width="25%", height="20%", loc='lower right',
                                bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=axes[1, 1].transAxes)
    legend_container.set_xlim(0, 1)
    legend_container.set_ylim(0, 1)
    legend_container.set_xticks([])
    legend_container.set_yticks([])

    # Add frame around the container
    legend_container.spines['top'].set_visible(True)
    legend_container.spines['right'].set_visible(True)
    legend_container.spines['bottom'].set_visible(True)
    legend_container.spines['left'].set_visible(True)
    legend_container.spines['top'].set_linewidth(1.5)
    legend_container.spines['right'].set_linewidth(1.5)
    legend_container.spines['bottom'].set_linewidth(1.5)
    legend_container.spines['left'].set_linewidth(1.5)
    legend_container.spines['top'].set_color('black')
    legend_container.spines['right'].set_color('black')
    legend_container.spines['bottom'].set_color('black')
    legend_container.spines['left'].set_color('black')

    # HF gradient bar (positioned in upper part of container)
    hf_gradient_ax = inset_axes(legend_container, width="80%", height="35%", loc='upper center',
                            bbox_to_anchor=(0, 0.1, 1, 0.8), bbox_transform=legend_container.transAxes)
    hf_gradient_ax.imshow(gradient_values, aspect='auto', cmap=hf_cmap_custom)
    hf_gradient_ax.set_xticks([])
    hf_gradient_ax.set_yticks([])
    hf_gradient_ax.set_title('HF calls ▷', fontsize=12, pad=4)

    # LF gradient bar (positioned in lower part of container)
    lf_gradient_ax = inset_axes(legend_container, width="80%", height="35%", loc='lower center',
                            bbox_to_anchor=(0, 0.0, 1, 0.8), bbox_transform=legend_container.transAxes)
    lf_gradient_ax.imshow(gradient_values, aspect='auto', cmap=lf_cmap_custom)
    lf_gradient_ax.set_xticks([])
    lf_gradient_ax.set_yticks([])
    lf_gradient_ax.set_title('LF calls ●', fontsize=12, pad=4)
    for ax in axes.flat:
        ax.grid(linestyle='--', alpha=0.6)
    return fig

# +
nhf_pair_loc = dw.loc.loc_from_picks(nhf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_pair_loc = dw.loc.loc_from_picks(nlf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
shf_pair_loc = dw.loc.loc_from_picks(shf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)
slf_pair_loc = dw.loc.loc_from_picks(slf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)

nhf_localizations = dw.loc.loc_from_picks(nhf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_localizations = dw.loc.loc_from_picks(nlf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
shf_localizations = dw.loc.loc_from_picks(shf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)
slf_localizations = dw.loc.loc_from_picks(slf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)

pair_assoc = (nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair)
pair_loc = (nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc)
associations = (nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list)
localizations = (nhf_localizations, nlf_localizations, shf_localizations, slf_localizations)
# -

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs)
fig.savefig('../figs/associations_bicable.pdf', bbox_inches=None, transparent=True)
plt.show()

# +
# test_n = nhf_assoc_list_pair[0]
# test_s = shf_assoc_list_pair[0]

n_test_loc = nhf_pair_loc[0]
s_test_loc = shf_pair_loc[0]

bicable_pos = (n_cable_pos, s_cable_pos)

def loc_picks_bicable(n_assoc, s_assoc, cable_pos, c0, fs, Nbiter=20):
    """
    Solve the least squares localization problem for a single cable using the picks' indices.
    
    Parameters
    ----------
    idx_dist : array-like
        The indices for the cable positions.
    idx_time : array-like
        The times corresponding to the cable positions.
    cable_pos : tuple
        A tuple containing the positions of the north and south cables.
    c0 : float
        The speed of sound or another relevant constant for localization.
    fs : float
        The sampling frequency.
    Nbiter : int, optional
        The number of iterations for the least squares solution, default is 20.
    
    Returns
    -------
    tuple
        A tuple containing the solution and the residuals of the least squares problem.
    """

    n_cable_pos, s_cable_pos = cable_pos
    bicable_pos = np.concatenate((n_cable_pos[n_assoc[0]], s_cable_pos[s_assoc[0]]))
    idx_time = np.concatenate((n_assoc[1], s_assoc[1]))
    idxmin_t = np.argmin(idx_time)  # Find the index of the minimum time

    times = idx_time / fs
    apex_loc = bicable_pos[idxmin_t, 0]  # Find the apex location from the minimum time index
    init = [apex_loc, np.mean(bicable_pos[:, 1]), -40, np.min(times)]  # Initial guess for the localization
    
    # Solve the least squares problem using the provided parameters
    n, residuals = dw.loc.solve_lq_weight(times, bicable_pos, c0, Nbiter, fix_z=True, ninit=init, residuals=True)
    
    return n, residuals

n_test, res = loc_picks_bicable(nhf_assoc_list_pair[0], shf_assoc_list_pair[0], bicable_pos, c0, fs)


def loc_picks_bicable_list(n_assoc_list, s_assoc_list, cable_pos, c0, fs, Nbiter=20):
    if len(n_assoc_list) != len(s_assoc_list):
        raise ValueError("The lengths of n_assoc_list and s_assoc_list must be equal.")

    localizations = []
    alt_localizations = []
    for i in tqdm(range(len(n_assoc_list))):
        n_assoc = n_assoc_list[i]
        s_assoc = s_assoc_list[i]
        n_loc, _ = loc_picks_bicable(n_assoc, s_assoc, cable_pos, c0, fs, Nbiter)
        localizations.append(n_loc)
        alt_loc, _ = loc_picks_bicable(n_assoc, s_assoc, cable_pos, c0, fs, Nbiter-1)
    return localizations, alt_localizations


# -

hf_pair_loc, hf_alt_loc = loc_picks_bicable_list(nhf_assoc_list_pair, shf_assoc_list_pair, bicable_pos, c0, fs)
lf_pair_loc, lf_alt_loc = loc_picks_bicable_list(nlf_assoc_list_pair, slf_assoc_list_pair, bicable_pos, c0, fs)

# +
# Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
opticald_n = []
opticald_s = []

for i in range(int(10000/2), len(df_north), int(10000/2)):
    opticald_n.append((df_north['x'][i], df_north['y'][i]))

for i in range(int(10000/2), len(df_south), int(10000/2)):
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

for i, loc in enumerate(hf_pair_loc):
    if i == 0:
        ax.plot(loc[0], loc[1], '>', c='tab:red', lw=4, label='Paired call - HF')
    ax.plot(loc[0], loc[1], '>', c='tab:red', lw=4)

for i, loc in enumerate(lf_pair_loc):
    if i == 0:
        ax.plot(loc[0], loc[1], 'o', c='tab:blue', lw=4, label='Paired call - LF')
    ax.plot(loc[0], loc[1], 'o', c='tab:blue', lw=4)

for i, loc in enumerate(nhf_localizations[:-2]):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], '^', c='tab:orange', label='Localized call - north - HF', lw=4)
    else:
        ax.plot(loc[0], loc[1], '^', c='tab:orange', lw=4)

for i, loc in enumerate(shf_localizations):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], 'v', c='tab:green', label='Localized call - south - HF', lw=4)
    else:
        ax.plot(loc[0], loc[1], 'v', c='tab:green', lw=4)
for i, loc in enumerate(nlf_localizations):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], '2', c='tab:green', label='Localized call - north - LF', lw=4)
    else:
        ax.plot(loc[0], loc[1], '2', c='tab:green', lw=4)
for i, loc in enumerate(slf_localizations):
    # Put label only for the first point
    if i == 0:
        ax.plot(loc[0], loc[1], '1', c='tab:orange', label='Localized call - south - LF', lw=4)
    else:
        ax.plot(loc[0], loc[1], '1', c='tab:orange', lw=4)

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
