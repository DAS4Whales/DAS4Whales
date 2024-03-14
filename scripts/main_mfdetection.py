# Imports
import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Download some DAS data

# %%
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

    # %% [markdown]
    # ### Get information on the DAS data from the hdf5 metadata

    # %%
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

    # %% [markdown]
    # ### Select the desired channels and channel interval

    # %%
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

    # %% [markdown]
    # ### Load raw DAS data
    # 
    # Loads the data using the pre-defined slected channels. 

    # %%
    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

    # %% [markdown]
    # ### Filtering in the frequency-wavenumber domain (f-k) and corresponding t-x plot

    # %%
    # Create the f-k filter 
    # includes band-pass filter trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1) 

    fk_filter = dw.dsp.hybrid_ninf_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
                                        cs_min=1350, cp_min=1450, cp_max=3300, cs_max=3450, fmin=14, fmax=30, display_filter=False)

    # Print the compression ratio given by the sparse matrix usage
    dw.tools.disp_comprate(fk_filter)
    # Apply the bandpass
    tr = dw.dsp.bp_filt(tr, fs, 14, 30)
    # Apply the f-k filter to the data, returns spatio-temporal strain matrix
    trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=False)

    # %%
    ## bandpass filter
    # trf_bp = dw.dsp.bp_filt(tr, fs, 14, 30)
    # trf_fk = dw.dsp.fk_filt(trf_bp, 1, fs, selected_channels[2], dx, 1350, 3450)

    # %%
    # # Plot the waterfall of the envelope 
    dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.4)
    ## Tests: 
    # test = np.max(abs(sp.hilbert(trf_fk, axis=1)) / np.std(trf_fk, axis=1, keepdims=True)) * 10 ** 9
    # dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1) / np.std(trf_fk, axis=1, keepdims=True), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=test * 0.1)
    # test = np.max(abs(sp.hilbert(trf_fk, axis=1))) * 10 ** 9
    # dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=test * 0.1)
    # test = np.max(trf_fk ** 2 / np.std(trf_fk, axis=1, keepdims=True) ** 2) * 10 ** 9
    # dw.plot.plot_tx(trf_fk ** 2 / np.std(trf_fk, axis=1, keepdims=True) ** 2, time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=test * 0.005)

    # %%
    # Get the indexes of the maximal value of the data:
    xi_m, tj_m = np.unravel_index(np.argmax(trf_fk, axis=None), trf_fk.shape)

    f, t, Sxx = sp.spectrogram(trf_fk[xi_m], metadata['fs'], nperseg=128, noverlap=0.95, scaling='spectrum', mode='magnitude', detrend=False)
    plt.figure(figsize=(12,4))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim([0, 35])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    dw.plot.plot_3calls(trf_fk[xi_m], time, 6.,27.6, 48.5)

    # %% [markdown]
    # ### Generate fin whale call template

    # %%
    tpl_paper = dw.detect.gen_template_fincall(time, fs, fmin = 15., fmax = 30., duration = 2.25, window=False)

    HF_note = dw.detect.gen_template_fincall(time, fs, fmin = 17.8, fmax = 28.8, duration = 0.68)
    LF_note = dw.detect.gen_template_fincall(time, fs, fmin = 14.7, fmax = 21.8, duration = 0.78)
    template = HF_note

    dw.plot.design_mf(trf_fk[xi_m], HF_note, LF_note, 6.17, 28., time, fs)

    # %% [markdown]
    # ### Compute the positive correlation matrix

    # %%
    print(np.std(HF_note))
    print(np.std(trf_fk[xi_m] / np.max(trf_fk[xi_m])))
    nf = int(6.17 * fs)
    print(33 / (len(HF_note) * np.std(HF_note) * np.std(trf_fk[xi_m] / np.max(trf_fk[xi_m]))))

    # %%
    # corr = sp.correlate(trf_fk[xi_m] / max(trf_fk[xi_m]), template, mode='full', method='fft')

    corr_m_HF = dw.detect.compute_cross_correlogram(trf_fk, HF_note)
    corr_m_LF = dw.detect.compute_cross_correlogram(trf_fk, LF_note)

    # %%
    # Plot the correlation matrix 

    maxv = max(np.max(corr_m_HF), np.max(corr_m_LF))
    fig = plt.figure(figsize=(16,8))
    plt.subplot(121)
    cplot = plt.imshow(abs(sp.hilbert(corr_m_HF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=maxv)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    plt.title('HF note', loc='right')

    plt.subplot(122)
    cplot = plt.imshow(abs(sp.hilbert(corr_m_LF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=maxv)
    bar = fig.colorbar(cplot, aspect=20)
    bar.set_label('cross-correlation []')
    plt.xlabel('Time [s]')
    plt.title('LF note', loc='right')
    plt.show()

    # %% [markdown]
    # ### Find the local maximas using find peaks and a threshold

    # %%
    print(f"The maximum correlation is {maxv}")
    thres = 0.5 * maxv
    print(thres)

    # %%
    # Find the arrival times and store them in a list of arrays format 
    peaks_indexes_m_HF = dw.detect.pick_times(corr_m_HF, fs, threshold=thres * 0.9)
    peaks_indexes_m_LF = dw.detect.pick_times(corr_m_LF, fs, threshold=thres)

    # Convert the list of array to tuple format
    peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_m_HF)
    peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_m_LF)

    # %% [markdown]
    # 

    # %% [markdown]
    # ### Plot the detection times over the strain matrix

    # %%
    dw.plot.detection_mf(trf_fk, peaks_indexes_tp_HF, peaks_indexes_tp_LF, time, dist, fs, dx, selected_channels, fileBeginTimeUTC)

    # %%
    # Finding the pick times for one trace
    corr = corr_m_HF[xi_m]
    lags = np.arange(0, len(trf_fk[xi_m])) / fs
    corr_pos = corr

    corr = abs(sp.hilbert(corr))
    peaks_indexes = sp.find_peaks(corr, prominence=0.5 * 0.8 * maxv)[0]

    plt.figure()
    plt.plot(lags, corr)
    plt.plot(lags[peaks_indexes], corr[peaks_indexes], 'ro', label='Peaks')
    plt.xlim(lags[0], lags[-1])
    plt.xlabel('lags+ [s]')
    plt.ylabel('crosscorr []')
    plt.xlim()
    plt.grid()
    plt.show()

    # %%
    plt.figure(figsize=(12, 4))
    plt.subplot(211)
    plt.plot(time, trf_fk[xi_m] / max(trf_fk[xi_m]), ls='-')
    plt.vlines(time[peaks_indexes], -1, 1, colors='r')
    plt.xlim([0., 60])
    plt.ylabel('normalized strain [-]')
    plt.xlabel('time [s]')
    plt.grid()

    plt.subplot(234)
    plt.plot(time, trf_fk[xi_m] / max(trf_fk[xi_m]))
    plt.vlines(time[peaks_indexes], -1, 1, colors='r')
    plt.ylabel('strain [-]')
    plt.xlabel('time [s]')
    plt.xlim([6., 8.])
    plt.grid()

    plt.subplot(235)
    plt.plot(time, trf_fk[xi_m] / max(trf_fk[xi_m]))
    plt.vlines(time[peaks_indexes], -1, 1, colors='r')
    plt.xlim([27.6, 29.6])
    plt.xlabel('time [s]')
    plt.grid()

    plt.subplot(236)
    plt.plot(time, trf_fk[xi_m] / max(trf_fk[xi_m]))
    plt.vlines(time[peaks_indexes], -1, 1, colors='r')
    plt.xlim([48.5, 50.5])
    plt.xlabel('time [s]')
    plt.grid()
    plt.tight_layout()

    plt.show()
    return

if __name__ == '__main__':
    main()


