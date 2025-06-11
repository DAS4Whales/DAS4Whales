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

# # Heatmap from the rectangular kernel density estimation

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
n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

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
peaks = (npeakshf, npeakslf, speakshf, speakslf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)

dw.assoc.plot_peaks(peaks, SNRs, selected_channels_m, dx, fs)
plt.show()

# +
# Choose to work on the HF or LF calls
# n_peaks = npeakshf
# s_peaks = speakshf
# n_SNR = nSNRhf
# s_SNR = sSNRhf
# print(s_peaks.shape)
# -

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
# -

# Parameters for the association process
dt_kde = 0.5 # [s] Time resolution of the KDE
bin_width = 1
# dt_kde = 0.25 # [s] Time resolution of the KDE (overlap)
# bin_width = 1.5
dt_tol = int(0.8 * fs) # [samples] Tolerance for the time index when removing picks
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
rms_threshold = 0.5
# Set the number of iterations for testing
iterations = 15

n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# +
# Precompute the time indices from peaks for both frequency bands and cables
n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs
n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs
s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs
s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs

# Calculate delayed picks for all grid points
n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

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
# Find the global min and max for KDE time range
all_delayed_picks = [n_delayed_picks_hf, n_delayed_picks_lf, s_delayed_picks_hf, s_delayed_picks_lf]
global_min = min(np.min(arr) for arr in all_delayed_picks)
global_max = max(np.max(arr) for arr in all_delayed_picks)

# Create time bins for KDE
Nkde = np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
t_kde = np.linspace(global_min, global_max, Nkde)

# Compute KDEs in parallel for each type
# North high frequency
n_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRhf) 
    for i in range(n_shape_x)
))

# North low frequency
n_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRlf)
    for i in range(n_shape_x)
))

# South high frequency
s_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRhf)
    for i in range(s_shape_x)
))

# South low frequency
s_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRlf)
    for i in range(s_shape_x)
))

# +
n_heatmap = np.max(n_kde_hf, axis=1)
s_heatmap = np.max(s_kde_hf, axis=1)

# Combined heatmap from summing the kdes
sum_kde_hf = n_kde_hf + s_kde_hf + n_kde_lf + s_kde_lf
# Hadamard product of the two kdes
prod_kde_hf = n_kde_hf * s_kde_hf * n_kde_lf * s_kde_lf

mu = np.mean(sum_kde_hf, axis=1)
sigma = np.std(sum_kde_hf, axis=1)

# -

idx = 0
alph = 1
plt.figure(figsize=(20, 8))
plt.plot(t_kde, sum_kde_hf[idx, :], label='North HF', color='tab:blue')
plt.hlines(y = mu[idx], xmin=t_kde[0], xmax=t_kde[-1], color='tab:blue', ls='--', label='Mean')
plt.hlines(y = sigma[idx], xmin=t_kde[0], xmax=t_kde[-1], color='tab:blue', ls='-.', label='Std')
plt.hlines(y = mu[idx] + alph*sigma[idx], xmin=t_kde[0], xmax=t_kde[-1], color='tab:blue', ls=':', label='Mean + 2*std')
plt.hlines(y = mu[idx] - alph*sigma[idx], xmin=t_kde[0], xmax=t_kde[-1], color='tab:blue', ls=':', label='Mean - 2*std')


# ## Plot the Heatmap for the north cable

mu_sp = np.mean(mu)
sigma_sp = np.std(sigma)
heatmap = np.max(prod_kde_hf, axis=1) 
fig = dw.assoc.plot_kdesurf(df_north, df_south, bathy, x, y, xg, yg, heatmap)
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
