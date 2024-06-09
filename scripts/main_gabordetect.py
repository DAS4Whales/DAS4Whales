# Gabor kernels augmented detection
# Imports
import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams['font.size'] = 20

def main(url):

    # Download some DAS data
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


    # ### Load raw DAS data
    # 
    # Loads the data using the pre-defined slected channels. 


    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)


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


    image = dw.improcess.trace2image(trf_fk)


    Nx = image.shape[0]
    Nt = image.shape[1]
    xpix = np.arange(0, Nx, 1)
    tpix = np.arange(0, Nt, 1)

    # Detection speed
    c0 = 1500 # m/s
    theta_c0 = dw.improcess.angle_fromspeed(c0, fs, dx, selected_channels)


    # Plot the matrix with slopes
    # plt.figure(figsize=(10, 6))
    # plt.imshow(image, aspect='equal', origin='lower', cmap='turbo')
    # plt.plot(xpix, tpix[:len(xpix)], 'w--')
    # plt.plot(xpix, ratio * tpix[:len(xpix)], 'r--')
    # plt.ylim([0, Nx])
    # plt.colorbar(label='Normalized amplitude', aspect=30, pad=0.015)
    # plt.xlabel('Time indices')
    # plt.ylabel('Distance indices')
    # plt.show()


    imagebin = dw.improcess.binning(image, 1/10, 1/10)


    gabfilt_up, gabfilt_down = dw.improcess.gabor_filt_design(theta_c0, plot=True)


    fimage = cv2.filter2D(imagebin, cv2.CV_64F, gabfilt_up) + cv2.filter2D(imagebin, cv2.CV_64F, gabfilt_down)


    plt.figure(figsize=(10, 6))
    plt.imshow(fimage, aspect='equal', origin='lower', cmap='turbo', vmin=0) #, vmin=-0.4 * np.max(np.abs(fimage)), vmax= 0.4 * np.max(np.abs(fimage)))
    plt.colorbar(label='Line feature score', aspect=30, pad=0.015)
    plt.xlabel('Time indices')
    plt.ylabel('Distance indices')
    plt.tight_layout()
    plt.show()

    # Threshold the image
    threshold = 9100  # 7800 for the first dataset
    binary_image = fimage > threshold

    plt.figure(figsize=(10, 6))
    plt.imshow(binary_image, aspect='equal', origin='lower', cmap='gray')
    plt.colorbar()
    plt.xlabel('Time indices')
    plt.ylabel('Distance indices')
    plt.tight_layout()
    plt.show()




    mask = cv2.filter2D(binary_image.astype(float), cv2.CV_64F, gabfilt_up) + cv2.filter2D(binary_image.astype(float), cv2.CV_64F, gabfilt_down)
    threshold2 = 150
    mask = mask > threshold2


    plt.figure(figsize=(10, 6))
    plt.imshow(mask, aspect='equal', origin='lower', cmap='gray')
    plt.colorbar()
    plt.xlabel('Time indices')
    plt.ylabel('Distance indices')
    plt.tight_layout()
    plt.show()



    # Smoothly apply the binary image as mask to the original image

    smoothed_image = dw.improcess.apply_smooth_mask(imagebin, mask)


    # plt.figure(figsize=(10, 6))
    # plt.imshow(smoothed_image, aspect='equal', origin='lower', cmap='turbo')
    # plt.colorbar(label='Normalized amplitude', aspect=30, pad=0.015)
    # plt.xlabel('Time indices')
    # plt.ylabel('Distance indices')
    # plt.show()



    ## Next: Try rezising the image to the original size
    # Sparse product of the mask with the original image
    mask_sparse = dw.improcess.binning(mask, 10, 10)

    # Apply the mask to the original trace
    masked_tr = dw.improcess.apply_smooth_mask(trf_fk, mask_sparse)
    dw.plot.plot_tx(sp.hilbert(masked_tr, axis=1), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.4)



    print(np.max(np.abs(masked_tr), axis=1))


    xi_m, tj_m = np.unravel_index(np.argmax(masked_tr, axis=None), trf_fk.shape)
    dw.plot.plot_3calls(masked_tr[xi_m], time, 6.,27.6, 48.5)
    # Spectrogram
    p,tt,ff = dw.dsp.get_spectrogram(masked_tr[xi_m+3000,:], fs, nfft=256, overlap_pct=0.95)
    dw.plot.plot_spectrogram(p, tt,ff, f_min = 10, f_max = 35, v_min=-45)


    # %matplotlib inline
    # import matplotlib.animation as animation
    # # Initialize variables and data
    # frames = 100 # np.shape(trf_fk)[0] - xi_m  # Number of frames to iterate over
    # fig, ax = plt.subplots(figsize=(12, 6))
    # roseus = dw.plot.import_roseus()
    # images = []
    # ax.set_title(f'Channel at {dist[xi_m] / 1e3 :.1f} km')
    # img = ax.pcolormesh([0,0,0], [0,0,0], [[0, 0], [0, 0]], cmap=roseus)
    # cbar = plt.colorbar(img, aspect=30, pad=0.015)
    # cbar.set_label('dB')

    # def update(i):
    #     ax.clear()
    #     p,tt,ff = dw.dsp.get_spectrogram(masked_tr[xi_m + 10 * i,:], fs, nfft=256, overlap_pct=0.95)
    #     shw = ax.pcolormesh(tt, ff, p, shading='auto', cmap=roseus, vmin=-45, vmax=None)
    #     ax.set_ylim(10, 35)
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Frequency (Hz)')
    #     ax.set_title(f'Channel at {dist[xi_m + 10 * i] / 1e3 :.1f} km')
    #     cbar.update_normal(shw)
    #     # cbar = fig.colorbar(img, aspect=20, pad=0.015, ax=[ax1, ax2], location='right')
    #     # cbar.set_label('Cross-correlation [-]')

    # # Create animation
    # ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=100)

    # # plt.colorbar(img, ax=[ax1, ax2], aspect=20, pad=0.015, location='right').set_label('Cross-correlation [-]')

    # from IPython.display import HTML
    # HTML(ani.to_html5_video())


    # Run the matched filter on the filtered data

    # Design the matched filter
    # HF_note = dw.detect.gen_template_fincall(time, fs, fmin = 17.8, fmax = 28.8, duration = 0.68)
    # LF_note = dw.detect.gen_template_fincall(time, fs, fmin = 14.7, fmax = 21.8, duration = 0.78)

    HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
    HF_note = np.hanning(len(HF_note)) * HF_note

    LF_note = dw.detect.gen_hyperbolic_chirp(14.7, 21.8, 0.78, fs)
    LF_note = np.hanning(len(LF_note)) * LF_note

    # plt.figure(figsize=(10, 6))
    # plt.plot(time, np.convolve(masked_tr[xi_m] / np.max(masked_tr[xi_m]), np.flip(HF_note), mode='same'))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    # LF_note = dw.detect.gen_template_fincall(time, fs, fmin = 14.7, fmax = 21.8, duration = 0.78)

    # Apply the matched filter
    from tqdm import tqdm
    # Compute correlation along axis 1
    cross_correlogram_HF = np.empty_like(masked_tr)
    cross_correlogram_LF = np.empty_like(masked_tr)

    for i in tqdm(range(masked_tr.shape[0])):
        if np.max(masked_tr[i, :]) > 0:
            cross_correlogram_HF[i, :] = sp.correlate(masked_tr[i, :] / np.max(masked_tr[i, :]), HF_note, mode='same', method='fft')
            cross_correlogram_LF[i, :] = sp.correlate(masked_tr[i, :] / np.max(masked_tr[i, :]), LF_note, mode='same', method='fft')
        


    # Plot the cross-correlograms
    # Plot the cross-correlograms
    maxv = max(np.max(cross_correlogram_HF), np.max(cross_correlogram_LF))

    dw.plot.plot_cross_correlogramHL(cross_correlogram_HF, cross_correlogram_LF, time, dist, maxv)


    # pick times of the calls
    # HF
    thres = 0.5 * maxv
    # Find the arrival times and store them in a list of arrays format 
    peaks_indexes_m_HF = dw.detect.pick_times_env(cross_correlogram_HF, threshold=thres * 0.9)
    peaks_indexes_m_LF = dw.detect.pick_times_env(cross_correlogram_LF, threshold=thres)

    # Convert the list of array to tuple format
    peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_m_HF)
    peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_m_LF)


    dw.plot.detection_mf(trf_fk, peaks_indexes_tp_HF, peaks_indexes_tp_LF, time, dist, fs, dx, selected_channels, fileBeginTimeUTC)
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

