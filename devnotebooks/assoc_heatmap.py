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

# +
# Load the peak indexes and the metadata
n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

# n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_08:00:02_ipi3_th_4.nc') 
# s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_08:00:02_ipi3_th_5.nc')

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
# # Sort the peaks based on SNR difference
npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
)

speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
)

# -

# # Plot the sorted peaks
peaks = (npeakshf, npeakslf, speakshf, speakslf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)
# dw.assoc.plot_peaks(peaks, SNRs, selected_channels_m, dx, fs)
dw.assoc.plot_tpicks_resolved(peaks, SNRs, selected_channels_m, dx, fs)
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
sum_kde = n_kde_hf + s_kde_hf + n_kde_lf + s_kde_lf
# Hadamard product of the two kdes
prod_kde = n_kde_hf * s_kde_hf * n_kde_lf * s_kde_lf

sum_north = n_kde_hf + n_kde_lf
sum_south = s_kde_hf + s_kde_lf
sum_hf = n_kde_hf + s_kde_hf
sum_lf = n_kde_lf + s_kde_lf

mu = np.mean(sum_kde)
mu_t = np.mean(sum_kde, axis=0)
mu_sp= np.mean(sum_kde, axis=1)

sigma = np.std(sum_kde)
sigma_t = np.std(sum_kde, axis=0)
sigma_sp = np.std(sum_kde, axis=1)

# Stats 
mu_north = np.mean(sum_north)
mu_south = np.mean(sum_south)
mu_north_t = np.mean(sum_north, axis=0)
mu_south_t = np.mean(sum_south, axis=0)
sigma_north = np.std(sum_north)
sigma_south = np.std(sum_south)
sigma_north_t = np.std(sum_north, axis=0)
sigma_south_t = np.std(sum_south, axis=0)

# Stats per call type
mu_hf = np.mean(sum_hf)
mu_lf = np.mean(sum_lf)
mu_hf_t = np.mean(sum_hf, axis=0)
mu_lf_t = np.mean(sum_lf, axis=0)
sigma_hf = np.std(sum_hf)
sigma_lf = np.std(sum_lf)
sigma_hf_t = np.std(sum_hf, axis=0)
sigma_lf_t = np.std(sum_lf, axis=0)

print(sum_kde.shape)


# +
# Same but sorting by call type
plt.rcParams.update({'font.size': 16})
# Shared parameters for north and south plots
# Create bins for the amplitude values
n_bins = 16
amp_min, amp_max = np.min(sum_hf), np.max(sum_hf)
vmin = 0.001
vmax = 1
vdelta = 0.05
beta = 3 # Number of stds 


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8), sharex=True, sharey=True, tight_layout=True)
ax1.set_title('HF calls')
# Create bins for the amplitude values
cbarticks = np.arange(vmin, vmax-vmin+vdelta, vdelta)
amp_bins = np.linspace(amp_min, amp_max, n_bins)

# Create 2D histogram (PDF) for each time point
pdf_hf = np.zeros((len(amp_bins)-1, len(t_kde)))

for i, t in enumerate(t_kde):
    hist, _ = np.histogram(sum_hf[:, i], bins=amp_bins, density=True)
    pdf_hf[:, i] = hist / np.sum(hist)  # Normalize the histogram to get Normalized spatial density

# Create contour plot of the PDF
T_mesh, A_mesh = np.meshgrid(t_kde, amp_bins[:-1])
levels = np.linspace(0, np.max(pdf_hf), 50)
im = ax1.contourf(T_mesh, A_mesh, pdf_hf, levels=cbarticks, cmap='viridis', norm=colors.Normalize(vmin=vmin, vmax=vmax))
# fig.colorbar(im, label='Normalized spatial density', ax=ax1)

# Plot percentiles over the contours
p5_hf = np.percentile(sum_hf, 5, axis=0)
p50_hf = np.percentile(sum_hf, 50, axis=0)  # median
p95_hf = np.percentile(sum_hf, 95, axis=0)

ax1.plot(t_kde, p50_hf, color='black', linewidth=2, label='50%')
ax1.plot(t_kde, p5_hf, color='black', linewidth=1, label='5%')
ax1.plot(t_kde, p95_hf, color='black', linewidth=1, label='95%')
ax1.plot(t_kde, mu_hf_t, color='red', label='$\\mu_t$')
ax1.hlines(mu_hf, t_kde[0], t_kde[-1], color='grey', linestyle='--', linewidth=3, label='$\\mu$')
ax1.hlines(mu_hf + beta * sigma_hf, t_kde[0], t_kde[-1], color='green', linestyle='--', linewidth=3, label=f'$\\mu + {beta} \\sigma$')
ax1.hlines(mu, t_kde[0], t_kde[-1], color='pink', linestyle='--', linewidth=3, label='$\\mu_g$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Occurrences [-]')
ax1.legend()

ax2.set_title('LF Calls')
# Create bins for the amplitude values
amp_min, amp_max = np.min(sum_lf), np.max(sum_lf)
amp_bins = np.linspace(amp_min, amp_max, n_bins)
# Create 2D histogram (PDF) for each time point
pdf_lf = np.zeros((len(amp_bins)-1, len(t_kde)))
for i, t in enumerate(t_kde):
    hist, _ = np.histogram(sum_lf[:, i], bins=amp_bins, density=True)
    pdf_lf[:, i] = hist / np.sum(hist)  # Normalize the histogram to get PDF
# Create contour plot of the PDF
T_mesh, A_mesh = np.meshgrid(t_kde, amp_bins[:-1])
levels = np.linspace(0, np.max(pdf_lf), 50)
im = ax2.contourf(T_mesh, A_mesh, pdf_lf, levels=cbarticks, cmap='viridis', norm=colors.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(im, label='Normalized spatial density', ax=ax2)

# Plot percentiles over the contours
p5_lf = np.percentile(sum_lf, 5, axis=0)
p50_lf = np.percentile(sum_lf, 50, axis=0)
p95_lf = np.percentile(sum_lf, 95, axis=0)
ax2.plot(t_kde, p50_lf, color='black', linewidth=2, label='50%')
ax2.plot(t_kde, p5_lf, color='black', linewidth=1, label='5%')
ax2.plot(t_kde, p95_lf, color='black', linewidth=1, label='95%')
ax2.plot(t_kde, mu_lf_t, color='red', label='$\\mu_t$')
ax2.hlines(mu_lf, t_kde[0], t_kde[-1], color='grey', linestyle='--', linewidth=3, label='$\\mu$')
ax2.hlines(mu_lf + beta * sigma_lf, t_kde[0], t_kde[-1], color='green', linestyle='--', linewidth=3, label=f'$\\mu + {beta} \\sigma$')
ax2.hlines(mu, t_kde[0], t_kde[-1], color='pink', linestyle='--', linewidth=3, label='$\\mu_g$')
ax2.set_xlabel('Time [s]')
# ax2.set_ylabel('Occurrences [-]')
ax2.set_xlim(t_kde[0], t_kde[-1])
ax2.set_ylim((0, 0.8 * max(np.max(sum_hf), np.max(sum_lf))))
ax2.legend()
plt.show()
# -

n_distances = np.sqrt((xg[:, None] - n_cable_pos[:, 0])**2 + (yg[:, None] - n_cable_pos[:, 1])**2 + 200*(zg - n_cable_pos[:, 2])**2)
s_distances = np.sqrt((xg[:, None] - s_cable_pos[:, 0])**2 + (yg[:, None] - s_cable_pos[:, 1])**2 + 200*(zg - s_cable_pos[:, 2])**2)
print(n_distances.shape, s_distances.shape)
distances = n_distances.min(axis=1) + s_distances.min(axis=1)
distances *= 0.5 # Average distances to both cables

# ## Plot the Heatmap for the north cable

# +
maxsum = np.max(sum_kde, axis=1)
maxprod = np.max(prod_kde, axis=1)
binary = np.ones_like(maxprod)

threshold = np.percentile(maxsum, 40)  # keep top 3%
binary[maxsum < threshold] = 0
fig = dw.assoc.plot_kdesurf(df_north, df_south, bathy, x, y, xg, yg, mu_sp+3*sigma_sp)
# fig = dw.assoc.plot_kdesurf(df_north, df_south, bathy, x, y, xg, yg, distances)
plt.show()

print(f'ratio of points above the threshold: {np.sum(binary) / binary.size:.2f}')

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
