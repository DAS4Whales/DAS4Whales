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

import das4whales as dw
import scipy.signal as sp
import scipy.fft as sfft
import numpy as np
import matplotlib.pyplot as plt     
import sparse
from datetime import datetime
import colorcet as cc

# +
# Functions 


def taper_data(trace):
    """
    Apply a Tukey window to each line (time series) of the input matrix.

    Parameters
    ----------
    trace : np.ndarray
        2D numpy array, where each column represents a time series.

    Returns
    -------
    np.ndarray
        Tapered matrix with the same shape as the input.
    """
    nt = trace.shape[1]
    # Change alpha to increase the tapering ratio
    trace *= sp.windows.tukey(nt, alpha=0.03)[np.newaxis, :]
    return trace


def fk_filter_sparsefilt(trace, fk_filter_matrix, tapering=False):
    """
    Applies a pre-calculated f-k filter to DAS strain data

    Parameters
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain.
    fk_filter_matrix : np.ndarray
        A [channel x time sample] nparray containing the f-k-filter.

    Returns
    -------
    np.ndarray
        A [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal domain.
    """
    if tapering:
        trace = taper_data(trace)

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    if isinstance(fk_filtered_trace, sparse.COO):
        # Convert the sparse matrix to a dense format
        fk_filtered_trace = fk_filtered_trace.todense()
    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real


def fk_filter_sparsefilt_par(trace, fk_filter_matrix, tapering=False):
    """
    Applies a pre-calculated f-k filter to DAS strain data

    Parameters
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain.
    fk_filter_matrix : np.ndarray or sparse.COO
        A [channel x time sample] nparray containing the f-k-filter.

    Returns
    -------
    np.ndarray
        A [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal domain.
    """
    if tapering:
        trace = taper_data(trace)
    
    trace = np.asarray(trace, dtype=np.complex64)

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(sfft.fft2(trace, workers=-1))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    if isinstance(fk_filtered_trace, sparse.COO):
        # Convert the sparse matrix to a dense format
        fk_filtered_trace = fk_filtered_trace.todense()
    # Back to the t-x domain
    trace = sfft.ifft2(np.fft.ifftshift(fk_filtered_trace), workers=-1)

    return trace.real


# -

# Matplotlib settings
plt.rcParams['font.size'] = 20

# +
url = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

filepath, filename = dw.data_handle.dl_file(url)

# Read HDF5 files and access metadata
# Get the acquisition parameters for the data folder
metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

print(f'Sampling frequency: {metadata["fs"]} Hz')
print(f'Channel spacing: {metadata["dx"]} m')
print(f'Gauge length: {metadata["GL"]} m')
print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
print(f'Number of channels: {metadata["nx"]}')
print(f'Number of time samples: {metadata["ns"]}')

# +
selected_channels_m = [20000, 40000, 3]
selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                     selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                           # channels along the FO Cable
                                           # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                           # numbers

print('Begin channel #:', selected_channels[0], 
      ', End channel #: ',selected_channels[1], 
      ', step: ',selected_channels[2], 
      'equivalent to ',selected_channels[2]*dx,' m')

# -

tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

# +
# Create the f-k filter 
# includes band-pass filter trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1) 

fk_params_s = {
    'c_min': 1400.,
    'c_max': 5000.,
    'fmin': 14.,
    'fmax': 28.
}

fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params_s, display_filter=False)
# -


print(type(fk_filter))
# Print fk_filter attributes
print(fk_filter.shape)
# Print the memory size of the filter
print(fk_filter.nbytes / 1e6, 'MB')
print(fk_filter.todense().nbytes / 1e6, 'MB (dense)')
print(fk_filter.dtype)


# +
# import time
# Measure the time taken for the fk_filter_sparsefilt function
# to = time.time()
# trf_fk = fk_filter_sparsefilt(tr, fk_filter, tapering=True)
# tf = time.time()
# print(f'Time taken for fk_filter_sparsefilt: {tf - to:.2f} seconds')

# t0 = time.time()
trf_fk_par = fk_filter_sparsefilt_par(tr, fk_filter, tapering=True)
# t1 = time.time()
# print(f'Time taken for fk_filter_sparsefilt_par: {t1 - t0:.2f} seconds')


# -

def plot_tx(trace, time, dist, title_time_info=0, fig_size=(12, 10), v_min=None, v_max=None, cbar_label='Strain Envelope (x$10^{-9}$)'):
    """
    Spatio-temporal representation (t-x plot) of the strain data

    Parameters:
    ----------
    trace : np.ndarray
        A [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    time : np.ndarray
        The corresponding time vector
    dist : np.ndarray
        The corresponding distance along the FO cable vector
    title_time_info : int, str, or datetime.datetime, optional
        A time reference to display or the plot title. Can be a UTC timestamp (int), 
        a formatted string, or a `datetime.datetime` object (default is 0).
    fig_size : tuple, optional
        Tuple of the figure dimensions (default is (12, 10))
    v_min : float, optional
        Sets the min nano strain amplitudes of the colorbar (default is None)
    v_max : float, optional
        Sets the max nano strain amplitudes of the colorbar (default is None)

    Returns:
    -------
    None

    Notes:
    ------
    This function plots a spatio-temporal representation (t-x plot) of the strain data. It uses the given strain data,
    time vector, and distance vector to create the plot. The plot shows the strain envelope as a color map, with time
    on the x-axis and distance on the y-axis. The color of each point in the plot represents the strain amplitude at
    that point. The function also supports customizing the figure size, colorbar limits, and title.

    """

    fig = plt.figure(figsize=fig_size)
    #TODO determine if the envelope should be implemented here rather than just abs
    # Replace abs(trace) per abs(sp.hilbert(trace, axis=1)) ? 
    shw = plt.imshow(np.abs(trace) * 1e9, extent=[time[0], time[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                     origin='lower', cmap=cc.cm.CET_L20, vmin=v_min, vmax=v_max)
    plt.ylabel('Distance (km)')
    plt.xlabel('Time [s]')
    bar = fig.colorbar(shw, aspect=30, pad=0.015)
    bar.set_label(cbar_label)
	
    if title_time_info:
        if isinstance(title_time_info, datetime):
            title_text = title_time_info.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(title_time_info, str):
            title_text = title_time_info
        elif isinstance(title_time_info, int):
            title_text = datetime.utcfromtimestamp(title_time_info).strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("title_time_info must be an int, str, or datetime.datetime.")
        plt.title(title_text, loc='right')
	
    plt.tight_layout()

    return fig




# +
fig = plot_tx(sp.hilbert(trf_fk_par, axis=1), time, dist, title_time_info='test', v_max=2)
# Change the x-axis limits to zoom in on a specific time range
ax = fig.gca()
ax.set_xlim([27.5, 30])  # Adjust the x-axis limits as needed
ax.set_ylim([25, 35])  # Adjust the y-axis limits as needed

# Tryout something else
# -

from scipy.ndimage import shift  # For fractional delay
def slant_stack(data, dx, fs, slowness):
    n_channels, n_samples = data.shape
    shifted = np.zeros_like(data)
    # for i in range(n_channels):
    #     delay = i * dx * slowness  # in seconds
    #     shift_samples = int(np.round(delay * fs))
    #     # Time shift: roll and zero pad to avoid wraparound
    #     shifted[i] = np.roll(data[i], -shift_samples)
    #     if shift_samples > 0:
    #         shifted[i, -shift_samples:] = 0
    #     elif shift_samples < 0:
    #         shifted[i, :-shift_samples] = 0
    # # Stack and compute envelope
    # beam = np.sum(shifted, axis=0)

    center = (n_channels - 1) / 2
    for i in range(n_channels):
            delay_sec = (i - center) * dx * slowness  # in seconds
            delay_samples = delay_sec * fs
            # Use fractional shift with interpolation
            shifted[i] = shift(data[i], -delay_samples, order=1, mode='constant', cval=0.0)

    beam = np.sum(shifted, axis=0)
    nonzero_counts = np.sum(np.abs(shifted) > 1e-12, axis=0)
    norm_beam = np.divide(
    beam,
    nonzero_counts,
    out=np.zeros_like(beam),
    where=nonzero_counts > 0
    )
    return norm_beam 




# +
best_p = 1/1700

def subarray_slant_stack(data, dx, fs, slownesses, window_size=20, step=1):
    n_channels, n_samples = data.shape
    results = []

    for start in range(0, n_channels - window_size + 1, step):
        sub_data = data[start:start+window_size]
        stacked = slant_stack(sub_data, dx, fs, slownesses)  # same as before
        results.append(stacked)  # shape: (n_slowness, n_samples)

    return np.array(results)

shifted = subarray_slant_stack(trf_fk_par, dx, fs, best_p)

plot_tx(sp.hilbert(shifted, axis=1), time, dist, title_time_info='Best Slowness: {:.5f} s/m'.format(best_p), v_max=0.6)
