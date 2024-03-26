# Comparison of fk-filters notebook

# Import the DAS4Whales module and dependencies
# Imports
import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

# matplotlib parameters
plt.rcParams['font.size'] = 20


def main(url):
        # Download some DAS data
        filepath = dw.data_handle.dl_file(url)
        # Read HDF5 files and access metadata
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


        # Load raw DAS data
        # Loads the data using the pre-defined slected channels. 

        tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

        # Filtering in the frequency-wavenumber domain (f-k) : filters comparison

        # input variables for filtering: 
        # band-pass filtering:
        fmin = 14.
        fmax = 30.
        # f-k filtering:
        csmin = 1350.
        cpmin = 1450.
        cpmax = 3300.
        csmax = 3450.

        ## 2 steps non infinite gaussian taper
        ## bandpass filter
        tr = dw.dsp.bp_filt(tr, fs, fmin, fmax)
        # trf_bp = tr

        ## non infinite gaussian taper from shima
        # trf_fk_nf_gs = dw.dsp.fk_filt(trf_bp, 1, fs, selected_channels[2], dx, 1350, 3450)

        ## non infinite hybrid cosine taper
        fk_filter_nf_cos = dw.dsp.hybrid_ninf_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, cs_min=csmin, cp_min=cpmin, cp_max=cpmax, cs_max=csmax, fmin=fmin, fmax=fmax,
        display_filter=False)

        trf_fk_nf_cos = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_nf_cos, tapering=False)

        ## non infinite hybrid gaussian taper
        fk_filter_nf_gs = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, cs_min=csmin, cp_min=cpmin, cp_max=cpmax, cs_max=csmax, fmin=fmin, fmax=fmax,
        display_filter=False)

        trf_fk_nf_gs = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_nf_gs, tapering=False)

        # includes band-pass filter trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1) 
        ## infinite hybrid cosine taper
        fk_filter_inf_cos = dw.dsp.hybrid_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
                                        cs_min=csmin, cp_min=cpmin, fmin=fmin, fmax=fmax, display_filter=False)
                                        
        trf_fk_inf_cos = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_inf_cos, tapering=False)

        ## infinite hybrid Gaussian taper
        fk_filter_inf_gs = dw.dsp.hybrid_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
                                        cs_min=csmin, cp_min=cpmin, fmin=fmin, fmax=fmax, display_filter=False)

        trf_fk_inf_gs = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_inf_gs, tapering=False)

        # Print the compression ratio given by the sparse matrix usage
        dw.tools.disp_comprate(fk_filter_inf_gs)

        # # Plot the waterfall of the envelope 
        # dw.plot.plot_tx(sp.hilbert(trf_fk_nf_gs, axis=1), time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.4)

        # Get the indexes of the maximal value of the data:
        xi_m, tj_m = np.unravel_index(np.argmax(trf_fk_inf_cos, axis=None), trf_fk_inf_cos.shape)
        plt.figure()
        plt.plot(dist / 1e3, trf_fk_inf_cos[:, tj_m])
        plt.ylabel('strain[-]')
        plt.xlabel('distance [km]')
        plt.xlim([26.4, 30.4])

        ## Compute the SNR matrix version
        # Infinite cosine tapers
        SNR_m = dw.dsp.snr_tr_array(trf_fk_inf_cos, env=True)  # 10 * np.log10(trf_fk_inf_cos ** 2 / np.std(trf_fk_inf_cos, axis=1, keepdims=True) ** 2)

        # Infinite gaussian tapers
        SNR_m2 = dw.dsp.snr_tr_array(trf_fk_inf_gs, env=True)

        # Non-infinite gaussian tapers
        SNR_m3 = dw.dsp.snr_tr_array(trf_fk_nf_gs, env=True)

        # Non-infinite cosine tapers
        SNR_m4 = dw.dsp.snr_tr_array(trf_fk_nf_cos, env=True)

        # Plot the SNR matrix 
        dw.plot.snr_matrix(SNR_m4, time, dist, 20, fileBeginTimeUTC)
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