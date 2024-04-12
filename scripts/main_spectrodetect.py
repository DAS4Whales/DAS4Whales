import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

def main(url):
    # Download some DAS data
    filepath = dw.data_handle.dl_file(url)

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

    # Select the desired channels and channel interval
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

    # Loads the data using the pre-defined slected channels. 
    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

    # Filtering in the frequency-wavenumber domain (f-k) and corresponding t-x plot
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
    # Plot the waterfall of the envelope 
    dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.4)

    # Call metrics from the OOI dataset calls 2021-11-04T020002 
    # HF call 
    f0_hf = 27
    f1_hf = 17.
    duration_hf = 0.8
    bandwidth = 4. # or 5?
    # LF call  
    f1_lf = 14. 
    f0_lf = 20. 
    duration_lf = 1.2
    # Get the indexes of the maximal value of the data:
    xi_m, tj_m = np.unravel_index(np.argmax(trf_fk, axis=None), trf_fk.shape)
    ## Spectrogram parameters
    # colormap
    roseus = dw.plot.import_roseus()
    # parameters
    window_size = 0.8 # in seconds, approx the size of a call
    overlap_pct = 0.95
    nperseg = int(window_size * fs)
    nhop = int(np.floor(nperseg * (1 - overlap_pct)))
    noverlap = nperseg - nhop
    print(f'nperseg: {nperseg}, noverlap: {noverlap}, hop_length: {nhop}')   
    # Frequency band of the cross-correlation (wider than kernel)
    fm_corr = f1_lf - 3 * bandwidth
    fM_corr = f0_lf + 3 * bandwidth

    # Generate fin whale call template
    # Inspired by the package [whaletracks](https://github.com/qgoestch/whaletracks/blob/main/whaletracks/detection/detect_calls.py) and the functions <br>
    # ```
    # detect_calls.plotwav()
    # detect_calls.buildkernel()
    # detect_calls.xcorr()
    # ```
    spectro, ff, tt = dw.detect.get_sliced_nspectrogram(trf_fk[xi_m], fs, fm_corr, fM_corr, nperseg, nhop, plotflag=True)
    tvec, fvec_sub, Finkernel_hf = dw.detect.buildkernel(f0_hf, f1_hf, bandwidth, duration_hf, ff ,tt, fs, fm_corr, fM_corr, plotflag=True)
    tvec, _, Finkernel_lf = dw.detect.buildkernel(f0_lf, f1_lf, bandwidth, duration_lf, ff ,tt, fs, fm_corr, fM_corr, plotflag=True)
    print(np.sum(Finkernel_hf), np.sum(Finkernel_lf))
    fs_spectro = spectro.shape[1] / tt[-1]
    testidx = int(26.5 * fs_spectro)
    spectro[:, testidx:testidx + len(tvec)] = Finkernel_lf
    fs_spectro = spectro.shape[1] / tt[-1]
    plt.figure(figsize=(12, 6))
    plt.imshow(spectro[:, testidx:testidx + int(6 * fs_spectro)], aspect='auto', origin='lower', cmap=roseus, extent=[tt[testidx], tt[testidx + int(6 * fs_spectro)], ff[0], ff[-1]], vmin=-0.5, vmax=1)
    int(10 * fs)

    
    flims = [fmin, fmax]
    kernelHF = {'f0': 27., 'f1': 17., 'dur': 0.8, 'bdwidth': 4.} 
    kernelLF = {'f0': 20., 'f1': 14., 'dur': 1.2, 'bdwidth': 4.}
    correlogram_HF = dw.detect.compute_cross_correlogram_spectrocorr(trf_fk, fs, flims, kernelHF, window_size, overlap_pct)
    correlogram_LF = dw.detect.compute_cross_correlogram_spectrocorr(trf_fk, fs, flims, kernelLF, window_size, overlap_pct)
    maxv = np.max((np.max(correlogram_HF), np.max(correlogram_LF))) 
    # dw.plot.plot_cross_correlogram(correlogram, time, dist, 0.5 * maxv)
    dw.plot.plot_cross_correlogramHL(correlogram_HF, correlogram_LF, time, dist, 0.5 * maxv)

    # Study SNR detection
    SNR_hf = dw.dsp.snr_tr_array(correlogram_HF, env=True)
    SNR_lf = dw.dsp.snr_tr_array(correlogram_LF, env=True)
    dw.plot.snr_matrix(SNR_hf, tt, dist, 20, fileBeginTimeUTC, '/spectrodetect: HF')
    dw.plot.snr_matrix(SNR_lf, tt, dist, 20, fileBeginTimeUTC, '/spectrodetect: LF')
    # Find the arrival times and store them in a list of arrays format 
    peaks_indexes_m_HF = dw.detect.pick_times(correlogram_HF, threshold = 14)
    peaks_indexes_m_LF = dw.detect.pick_times(correlogram_LF, threshold = 14)
    # Convert the list of array to tuple format
    peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_m_HF)
    peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_m_LF)
    spectro_fs = np.shape(correlogram_HF)[1] / time[-1]
    dw.plot.detection_spectcorr(trf_fk, peaks_indexes_tp_HF, peaks_indexes_tp_LF, time, dist, spectro_fs, dx, selected_channels, file_begin_time_utc=fileBeginTimeUTC)

    return

if __name__ == "__main__":
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