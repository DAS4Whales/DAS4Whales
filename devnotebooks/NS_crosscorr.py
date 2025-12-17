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

# # North-South cross-correlation to get delays

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
n_ds = xr.load_dataset('../out/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
s_ds = xr.load_dataset('../out/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

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

n_times = n_ds.attrs['data_shape'][1]
s_times = s_ds.attrs['data_shape'][1]
# -

dt_tol = int(0.5 * fs)  # 0.5 s
dist_tol = int(10/dx)
print('dt_tol: ', dt_tol)
print('dist_tol: ', dist_tol)

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
speakshf = speakshf[:, sSNRhf > 5]
speakslf = speakslf[:, sSNRlf > 5]

sSNRhf = sSNRhf[sSNRhf > 5]
sSNRlf = sSNRlf[sSNRlf > 5]
peaks = (npeakshf, npeakslf, speakshf, speakslf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)

dw.assoc.plot_peaks(peaks, SNRs, selected_channels_m, dx, fs)
plt.show()

# +
# Just plot north and south hf 
nhf_dist = n_selected_channels[0] + npeakshf[0][:] * dx
nhf_times = npeakshf[1][:] / fs

shf_dist = s_selected_channels[0] + speakshf[0][:] * dx
shf_times = speakshf[1][:] / fs


print(len(nhf_dist), len(nhf_times), len(nSNRhf))
print(len(shf_dist), len(shf_times), len(sSNRhf))

plt.figure()
plt.scatter(nhf_times, nhf_dist * 1e-3, c=nSNRhf, cmap='viridis', s=nSNRhf)
plt.colorbar(label='SNR')
plt.xlabel('Time (s)')
plt.ylabel('Distance (km)')
plt.show()

plt.figure()
plt.scatter(shf_times, shf_dist * 1e-3, c=sSNRhf, cmap='viridis', s=sSNRhf)
plt.colorbar(label='SNR')
plt.xlabel('Time (s)')
plt.ylabel('Distance (km)')
plt.show()

# +
# Recreate and array with SNRs and zeros for both cables
nSNRs_array = np.zeros((len(n_dist), n_times), dtype=np.float32)
sSNRs_array = np.zeros((len(s_dist), s_times), dtype=np.float32)



# +
import numpy as np
from scipy import signal
from scipy.stats import binned_statistic_2d

def grid_2d_cross_correlation(n_times, n_dist, n_snr, s_times, s_dist, s_snr, 
                             time_bins=200, dist_bins=200):
    """
    Grid both datasets in 2D (time-distance) space and cross-correlate
    """
    # Define common grid bounds
    t_min = max(n_times.min(), s_times.min())
    t_max = min(n_times.max(), s_times.max())
    d_min = max(n_dist.min(), s_dist.min())
    d_max = min(n_dist.max(), s_dist.max())
    
    # Create 2D grids
    time_edges = np.linspace(t_min, t_max, time_bins + 1)
    dist_edges = np.linspace(d_min, d_max, dist_bins + 1)
    
    # Bin the data (using SNR as values, could also use event counts)
    n_grid, _, _, _ = binned_statistic_2d(n_times, n_dist, n_snr, 
                                         bins=[time_edges, dist_edges], 
                                         statistic='mean')
    s_grid, _, _, _ = binned_statistic_2d(s_times, s_dist, s_snr, 
                                         bins=[time_edges, dist_edges], 
                                         statistic='mean')
    
    # Replace NaN with zeros for correlation
    n_grid = np.nan_to_num(n_grid, nan=0.0)
    s_grid = np.nan_to_num(s_grid, nan=0.0)
    
    # 2D cross-correlation
    correlation = signal.correlate2d(n_grid, s_grid, mode='full')
    
    return correlation, n_grid, s_grid, time_edges, dist_edges

# Run 2D correlation
correlation, n_grid, s_grid, time_edges, dist_edges = grid_2d_cross_correlation(
    nhf_times, nhf_dist, nSNRhf, shf_times, shf_dist, sSNRhf)


# +

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# North grid
im1 = axes[0,0].imshow(n_grid.T, extent=[time_edges[0], time_edges[-1], 
                                        dist_edges[0]*1e-3, dist_edges[-1]*1e-3], 
                       aspect='auto', origin='lower', cmap='viridis')
axes[0,0].set_title('North Array Grid')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Distance (km)')
plt.colorbar(im1, ax=axes[0,0])

# South grid
im2 = axes[0,1].imshow(s_grid.T, extent=[time_edges[0], time_edges[-1], 
                                        dist_edges[0]*1e-3, dist_edges[-1]*1e-3], 
                       aspect='auto', origin='lower', cmap='viridis')
axes[0,1].set_title('South Array Grid')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Distance (km)')
plt.colorbar(im2, ax=axes[0,1])

# Cross-correlation
im3 = axes[1,0].imshow(correlation, aspect='auto', origin='lower', cmap='viridis', extent=[
    -time_edges[-1], time_edges[-1], -dist_edges[-1]*1e-3, dist_edges[-1]*1e-3])
axes[1,0].set_title('2D Cross-correlation')
axes[1,0].set_xlabel('Time Lag (s)')
axes[1,0].set_ylabel('Distance Lag (km)')
plt.colorbar(im3, ax=axes[1,0])

# Correlation peak
max_corr_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
axes[1,1].plot(correlation[max_corr_idx[0], :], label='Max correlation slice')
axes[1,1].set_title('Cross-correlation at Peak')
axes[1,1].set_xlabel('Distance Lag (bins)')
axes[1,1].set_ylabel('Correlation')
axes[1,1].legend()

plt.tight_layout()
plt.show()
