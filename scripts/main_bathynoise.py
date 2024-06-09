
#Import the DAS4Whales module and dependencies

# Imports
import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
from scipy.ndimage import convolve
from datetime import datetime
import torchvision.transforms as transforms
import pandas as pd
plt.rcParams['font.size'] = 20

# Download some DAS data

# The dataset of this example is constituted of 60s time series along  66 km of cable
url_before = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T015902Z.h5'

url = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

url_next = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020102Z.h5'

def main(url):

    filepath = dw.data_handle.dl_file(url)

    # ### Get information on the DAS data from the hdf5 metadata

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

    # ### Select the desired channels and channel interval

    selected_channels_m = [20000, 65000, 5]  # list of values in meters corresponding to the starting,
                                            # ending and step wanted channels along the FO Cable
                                            # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                            # in meters

    selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                        selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                            # channels along the FO Cable
                                            # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                            # numbers

    print('Begin channel #:', selected_channels[0], 
        ', End channel #: ',selected_channels[1], 
        ', step: ',selected_channels[2], 
        'equivalent to ',selected_channels[2]*dx,' m')

    # Load the bathymetry

    # load the .txt file and create a pandas dataframe
    df = pd.read_csv('data/north_DAS_latlondepth.txt', delimiter = ",", header = None)
    df.columns = ['chan_idx','lat', 'lon', 'depth']
    chan_m = df['chan_idx'] * dx
    df

    # Load raw DAS data
    # 
    # Loads the data using the pre-defined slected channels. 

    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

    ## Plot the raw data

    fig = plt.figure(figsize=(12, 10))
    wv = plt.imshow(tr * 1e9, aspect='auto', cmap='RdBu', extent=[min(time),max(time),min(dist)*1e-3,max(dist)*1e-3], origin='lower', vmin=-650, vmax=650)
    plt.title('Raw DAS data')
    plt.ylabel('Distance [km]')
    plt.xlabel('Time [s]')
    bar = fig.colorbar(wv, aspect=30, pad=0.015)
    bar.set_label(label='Strain [-] (x$10^{-9}$)')
    plt.show()


    ## Plot the raw data with the bathymetry 
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
    plt.rc('font', size=20) 
    plt.rc('xtick', labelsize=16)  
    plt.rc('ytick', labelsize=16)

    ax1 = plt.subplot(gs[1])
    wv = ax1.imshow(tr * 1e9, aspect='auto', cmap='RdBu', extent=[min(time),max(time),min(dist)*1e-3,max(dist)*1e-3], origin='lower', vmin=-600, vmax=600)
    ax1.set_title('Raw DAS data')
    ax1.set_ylabel('Distance [km]')
    ax1.set_xlabel('Time [s]')
    bar = fig.colorbar(wv, aspect=30, pad=0.015)
    bar.set_label(label='Strain [-] (x$10^{-9}$)')

    ax2 = plt.subplot(gs[0], sharey=ax1)
    ax2.plot(df['depth'], df['chan_idx'] * dx /1e3)
    ax2.set_xlabel('Depth [m]')
    ax2.set_ylim(20, 65)
    ax2.set_xlim(-550,-100)
    ax2.invert_xaxis()
    ax2.yaxis.tick_right()
    ax2.grid()

    plt.tight_layout()
    plt.show()

    # Create the f-k filter 
    # includes band-pass filter trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1) 
    fmin = 14
    fmax = 30
    fk_filter = dw.dsp.hybrid_ninf_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
                                        cs_min=1350, cp_min=1450, cp_max=3300, cs_max=3450, fmin=fmin, fmax=fmax, display_filter=False)

    # Print the compression ratio given by the sparse matrix usage
    dw.tools.disp_comprate(fk_filter)
    # Apply the bandpass
    tr = dw.dsp.bp_filt(tr, fs, fmin, fmax)
    # Apply the f-k filter to the data, returns spatio-temporal strain matrix
    trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=False)
    del tr

    # # Tryout of torch binning

    image = np.abs(sp.hilbert(trf_fk * 1e9, axis=1)) / np.std(trf_fk * 1e9, axis=1, keepdims=True)

    def plot_tx_noise(trace, time, dist,tnoise, file_begin_time_utc=0, fig_size=(12, 10), v_min=None, v_max=None):
        """
        Spatio-temporal representation (t-x plot) of the strain data

        Inputs:
        :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
        :param time: the corresponding time vector
        :param dist: the corresponding distance along the FO cable vector
        :param file_begin_time_utc: the time stamp of the represented file
        :param fig_size: Tuple of the figure dimensions. Default fig_size=(12, 10)
        :param v_min: sets the min nano strain amplitudes of the colorbar. Default v_min=0
        :param v_max: sets the max nano strain amplitudes of the colorbar, Default v_max=0.2

        Outputs:
        :return: a tx plot

        """
        t0, t1 = tnoise
        fig = plt.figure(figsize=fig_size)
        shw = plt.imshow(abs(trace), extent=[time[0], time[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                        origin='lower', cmap='turbo', vmin=v_min, vmax=v_max)
        plt.vlines(t0, dist[0] * 1e-3, dist[-1] * 1e-3, color='r', linestyles='dotted')
        plt.vlines(t1, dist[0] * 1e-3, dist[-1] * 1e-3, color='r', linestyles='dotted')
        plt.ylabel('Distance (km)')
        plt.xlabel('Time (s)')
        bar = fig.colorbar(shw, aspect=30, pad=0.015)
        bar.set_label('Strain Envelope (x$10^{-9}$)')

        if isinstance(file_begin_time_utc, datetime):
            plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')
        plt.tight_layout()
        plt.show()


    def norm(trace):
        """
        Normalize the data by its standard deviation
        """
        return trace / np.std(trace, axis=1, keepdims=True)

    # Analyse the channels
    # median of envelope
    med = np.median(abs(sp.hilbert(trf_fk, axis=1)), axis=1)
    # mean of envelope
    mean = np.mean(abs(sp.hilbert(trf_fk, axis=1)), axis=1)
    # standard deviation of envelope
    # std_env = np.std(abs(sp.hilbert(trf_fk, axis=1)), axis=1)
    # standard deviation of trace   
    std = np.std(trf_fk, axis=1)

    # difference between the standard deviation and the median of the envelope
    std_med_diff = std - med

    SNR_1d = 20 * np.log10(std / med)

    # Define the x ranges for shading
    x_ranges = [(26.5, 30.5), (46, 52)]

    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(dist / 1e3, med, label='Median of envelope')
    # plt.plot(dist / 1e3, mean, label='Mean of envelope')
    # plt.plot(dist / 1e3, std_env, label='Standard deviation of envelope')
    plt.plot(dist / 1e3, std, label='Standard deviation')
    plt.plot(dist / 1e3, std_med_diff, label='Std - Median of envelope', ls='--')
    plt.xlabel('Distance [km]')
    plt.ylabel('Strain [-]')
    plt.xlim([dist[0] / 1e3, dist[-1] / 1e3])
    plt.ylim([0, max(std) * 1.1])

    # Shade the background in the specified x ranges
    for x_range in x_ranges:
        plt.fill_between(x_range, plt.ylim()[0], plt.ylim()[1], color='gray', alpha=0.3)

    # Label the shaded areas
    for i, x_range in enumerate(x_ranges):
        x_start, x_end = x_range
        y_pos = plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1
        plt.text((x_start + x_end) / 2, y_pos, f'Apex {i+1}', ha='center')

    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dist / 1e3, SNR_1d)
    plt.xlabel('Distance [km]')
    plt.ylabel('Std to median ratio [dB]')
    plt.xlim([dist[0] / 1e3, dist[-1] / 1e3])
    plt.ylim([0, max(SNR_1d) * 1.1])

    # Shade the background in the specified x ranges
    for x_range in x_ranges:
        plt.fill_between(x_range, plt.ylim()[0], plt.ylim()[1], color='gray', alpha=0.3)

    # Label the shaded areas
    for i, x_range in enumerate(x_ranges):
        x_start, x_end = x_range
        y_pos = plt.ylim()[1] - (plt.ylim()[1] - plt.ylim()[0]) * 0.1
        plt.text((x_start + x_end) / 2, y_pos, f'Apex {i+1}', ha='center')

    plt.grid()
    plt.tight_layout()
    plt.show()



    tnoise1 = [19, 26]
    tnoise2 = [15, 19]
    dlim = 45
    idx_noise = [int(t * fs) for t in tnoise1]
    # Plot the waterfall of the envelope 
    # env = norm(abs(sp.hilbert(trf_fk, axis=1)))
    plot_tx_noise(image, time, dist, tnoise2, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=15)

    noise = trf_fk[:, idx_noise[0]:idx_noise[1]]
    noise_power = np.mean(noise**2, axis=1)
    noise_power_db = 10 * np.log10(noise_power/1e-11**2)
    noise_mean = np.mean(np.abs(sp.hilbert(noise, axis=1)), axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(dist / 1e3, noise_power_db)
    plt.xlabel('Distance [km]')
    plt.ylabel('Noise power [dB]')
    plt.grid()
    plt.show()

    return


if __name__ == '__main__':
    # The dataset of this example is constituted of 60s time series along  66 km of cable
    url_before = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T015902Z.h5'

    url = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

    url_next = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
        'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020102Z.h5'
    main(url)

