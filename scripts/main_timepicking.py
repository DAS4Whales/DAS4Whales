
# Libraries import
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import das4whales as dw
import cv2
import gc
from tqdm import tqdm

# Comment out these lines to enable the plots to display
import matplotlib
# matplotlib.use('Agg')

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelpad'] = 20


def main(urls, selected_channels_m):
        # North cable plots
        if len(urls) == 1: 
                # Download some DAS data
                url = urls[0]
                filepath, filename = dw.data_handle.dl_file(url)

                # Read HDF5 files and access metadata
                # Get the acquisition parameters for the data folder
                metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
                metadata["cablename"] = filename.split('-')[0]
                fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

                print(f'Sampling frequency: {metadata["fs"]} Hz')
                print(f'Channel spacing: {metadata["dx"]} m')
                print(f'Gauge length: {metadata["GL"]} m')
                print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
                print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
                print(f'Number of channels: {metadata["nx"]}')
                print(f'Number of time samples: {metadata["ns"]}')


                # ### Select the desired channels and channel interval

                selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                                selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                                        # channels along the FO Cable
                                                        # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                                        # numbers

                print('Begin channel #:', selected_channels[0], 
                ', End channel #: ',selected_channels[1], 
                ', step: ',selected_channels[2], 
                'equivalent to ',selected_channels[2]*dx,' m')


                ### Load raw DAS data
                
                # Loads the data using the pre-defined selected channels. 

                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
                metadata["fileBeginTimeUTC"] = fileBeginTimeUTC.strftime("%Y-%m-%d_%H:%M:%S")
        # South cable plots
        else:
                # Download the DAS data
                filepaths = []
                filenames = []
                for url in urls:
                        print(url)
                        filepath, filename = dw.data_handle.dl_file(url)
                        filepaths.append(filepath)
                        filenames.append(filename)

                metadata = dw.data_handle.get_acquisition_parameters(filepaths[0], interrogator='optasense')
                metadata["cablename"] = filenames[0].split('-')[0]
                fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

                selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, dx)

                timestamp = '2021-11-04 02:00:02.025000'
                duration = 60
                selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, dx)

                # Load the data
                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_mtpl_das_data(filepaths, selected_channels, metadata, timestamp, duration)
                metadata["fileBeginTimeUTC"] = fileBeginTimeUTC.strftime("%Y-%m-%d_%H:%M:%S")

        # Create the f-k filters
        fk_params = {   # Parameters for the signal
        'c_min': 1400.,
        'c_max': 3300.,
        'fmin': 14.,
        'fmax': 30.
        }

        fk_params_n = {   # Parameters for the noise
        'c_min': 1400.,
        'c_max': 3300.,
        'fmin': 32.,
        'fmax': 48.
        }

        fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=False)
        fk_filter_noise = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params_n, display_filter=False)

        # Apply the f-k filter to the data, returns spatio-temporal strain matrix
        trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
        noise = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_noise, tapering=True)

        noise = dw.dsp.normalize_std(noise)
        window_size = 100
        noise = dw.dsp.moving_average_matrix(abs(sp.hilbert(noise, axis=1)), window_size)

        # Delete the raw data to free memory
        del tr

        # Create the matched filters for detection
        HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
        HF_note = np.hanning(len(HF_note)) * HF_note

        LF_note = dw.detect.gen_hyperbolic_chirp(14.7, 21.8, 0.78, fs)
        LF_note = np.hanning(len(LF_note)) * LF_note

        # Apply the matched filter to the data 
        nmf_m_HF = dw.detect.calc_nmf_correlogram(trf_fk, HF_note)
        nmf_m_LF = dw.detect.calc_nmf_correlogram(trf_fk, LF_note)

        # Free memory
        del trf_fk
        gc.collect()

        ######  Perform the time picking on the matched filtered traces ######
        SNR_hfn = dw.dsp.snr_tr_array(nmf_m_HF)
        SNR_lfn = dw.dsp.snr_tr_array(nmf_m_LF)

        SNR_hfn = cv2.GaussianBlur(SNR_hfn, (9, 73), 0)
        SNR_lfn = cv2.GaussianBlur(SNR_lfn, (9, 73), 0)

        # Threshold the SNR matrix in an efficient way
        SNR_hfn = np.where(SNR_hfn < 0, 0, SNR_hfn)
        SNR_lfn = np.where(SNR_lfn < 0, 0, SNR_lfn)

        ipi = 2 # Inter pulse interval in seconds
        th = 4. # Threshold for the peak detection in dB

        peaks_indexes_HF = []
        peaks_indexes_LF = []

        for corr in tqdm(SNR_hfn, desc="Picking times"):
                peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=th)
                peaks_indexes_HF.append(peaks_indexes)

        for corr in tqdm(SNR_lfn, desc="Picking times"):
                peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=th)  
                peaks_indexes_LF.append(peaks_indexes)

        peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
        peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

        print('HF detections before denoising:', len(peaks_indexes_tp_HF[0]))
        print('LF detections before denoising:', len(peaks_indexes_tp_LF[0]))

        cable = filename.split('-')[0]
        if cable == 'North':
                fignum = 28
        if cable == 'South':
                fignum = 32

        # Sorth the sizes of the picks by SNR
        sizes_hf = SNR_hfn[peaks_indexes_tp_HF[0], peaks_indexes_tp_HF[1]]
        sizes_lf = SNR_lfn[peaks_indexes_tp_LF[0], peaks_indexes_tp_LF[1]]

        # Scale the sizes of the picks
        max_size = 140
        min_size = 2

        # Scale the sizes to the range [0, 1]
        sizes_hf_scaled = min_size + (sizes_hf - np.min(sizes_hf)) / (np.max(sizes_hf) - np.min(sizes_hf)) * (max_size - min_size)
        sizes_lf_scaled = min_size + (sizes_lf - np.min(sizes_lf)) / (np.max(sizes_lf) - np.min(sizes_lf)) * (max_size - min_size)

        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_HF[1] / fs, (peaks_indexes_tp_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='tab:blue', marker='.', s=sizes_hf_scaled, rasterized=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'HF note picks, {metadata["cablename"]}', loc='right')

        # Legend
        # Create a set of legend handles with different sizes
        handles = [
        plt.scatter([], [], s=min(sizes_hf_scaled), color='tab:blue', label=f'Min SNR: {sizes_hf.min():.1f}'),
        plt.scatter([], [], s=(min(sizes_hf_scaled) + max(sizes_hf_scaled)) / 2, color='tab:blue', label=f'Mid SNR: {np.median(sizes_hf):.1f}'),
        plt.scatter([], [], s=max(sizes_hf_scaled), color='tab:blue', label=f'Max SNR: {sizes_hf.max():.1f}')
        ]

        plt.legend(handles=handles, title="SNR Sizes", scatterpoints=1, loc='upper right')
        plt.savefig(f"figs/Figure_{fignum}.pdf")
        fignum += 1
        # plt.show()


        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_LF[1] / fs, (peaks_indexes_tp_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='tab:red', marker='.', s=sizes_lf_scaled, rasterized=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'LF note picks, {metadata["cablename"]}', loc='right')

        # Legend
        # Create a set of legend handles with different sizes
        handles = [
        plt.scatter([], [], s=min(sizes_lf_scaled), color='tab:red', label=f'Min SNR: {sizes_lf.min():.1f}'),
        plt.scatter([], [], s=(min(sizes_lf_scaled) + max(sizes_lf_scaled)) / 2, color='tab:red', label=f'Mid SNR: {np.median(sizes_lf):.1f}'),
        plt.scatter([], [], s=max(sizes_lf_scaled), color='tab:red', label=f'Max SNR: {sizes_lf.max():.1f}')
        ]

        plt.legend(handles=handles, title="SNR Sizes", scatterpoints=1, loc='upper right')
        plt.savefig(f"figs/Figure_{fignum}.pdf")
        fignum += 1
        # plt.show()

        # Make a count of the detections depending on IPI and threshold
        thresholds = np.arange(2, 20.5, 0.5) # Thresholds in dB
        IPIs = np.arange(0, 5.5, 0.5) # Inter pulse intervals in seconds
        IPIs[0] = 1/fs # Force the first IPI to be the sampling period
        detect_cptHFnoise = [[], []]
        detect_cptLFnoise = [[], []]

        for t in tqdm(thresholds, desc=f"Picking times, iterating over thresholds"):
                peaks_indexes_HF = []
                peaks_indexes_LF = []

                for corr in SNR_hfn:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=t)
                        peaks_indexes_HF.append(peaks_indexes)

                for corr in SNR_lfn:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=t)  
                        peaks_indexes_LF.append(peaks_indexes)

                peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
                peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

                detect_cptHFnoise[0].append(len(peaks_indexes_tp_HF[0]))
                detect_cptLFnoise[0].append(len(peaks_indexes_tp_LF[0]))

        for it in tqdm(IPIs, desc=f"Picking times, iterating over IPIs"):
                peaks_indexes_HF = []
                peaks_indexes_LF = []

                for corr in SNR_hfn:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = it * fs, height=th)
                        peaks_indexes_HF.append(peaks_indexes)

                for corr in SNR_lfn:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = it * fs, height=th)  
                        peaks_indexes_LF.append(peaks_indexes)

                peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
                peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

                detect_cptHFnoise[1].append(len(peaks_indexes_tp_HF[0]))
                detect_cptLFnoise[1].append(len(peaks_indexes_tp_LF[0]))


        # Free memory
        del SNR_lfn, SNR_hfn, peaks_indexes_HF, peaks_indexes_LF, peaks_indexes_tp_HF, peaks_indexes_tp_LF
        gc.collect()
        
        ########  Perform the time picking on the denoised mf traces #########

        # Normalize the matched filtered traces
        nmf_m_HF = dw.dsp.normalize_std(nmf_m_HF)
        nmf_m_LF = dw.dsp.normalize_std(nmf_m_LF)

        # Plot the SNR of the matched filter
        SNR_hf = 20 * np.log10(abs(sp.hilbert(nmf_m_HF, axis=1)) / abs(sp.hilbert(noise, axis=1)))
        SNR_lf = 20 * np.log10(abs(sp.hilbert(nmf_m_LF, axis=1)) / abs(sp.hilbert(noise, axis=1)))

        # Free memory
        del nmf_m_HF, nmf_m_LF, noise
        gc.collect()

        SNR_hf = cv2.GaussianBlur(SNR_hf, (9, 73), 0)
        SNR_lf = cv2.GaussianBlur(SNR_lf, (9, 73), 0)

        # Threshold the SNR matrix in an efficient way
        SNR_hf = np.where(SNR_hf < 0, 0, SNR_hf)
        SNR_lf = np.where(SNR_lf < 0, 0, SNR_lf)

        peaks_indexes_HF = []
        peaks_indexes_LF = []

        for corr in tqdm(SNR_hf, desc="Picking times"):
                peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=th)
                peaks_indexes_HF.append(peaks_indexes)

        for corr in tqdm(SNR_lf, desc="Picking times"):
                peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=th)  
                peaks_indexes_LF.append(peaks_indexes)

        peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
        peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

        # Save the time picking results
        # np.save(f'out/peaks_indexes_tp_HF_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}_ipi{ipi}_th_{th}.npy', peaks_indexes_tp_HF)
        # np.save(f'out/peaks_indexes_tp_LF_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}_ipi{ipi}_th_{th}.npy', peaks_indexes_tp_LF)

        # # Save the SNR matrixes 
        # np.save(f'out/SNR_hf_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}.npy', SNR_hf)
        # np.save(f'out/SNR_lf_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}.npy', SNR_lf)

        print('HF detections after denoising:', len(peaks_indexes_tp_HF[0]))
        print('LF detections after denoising:', len(peaks_indexes_tp_LF[0]))

                # Sorth the sizes of the picks by SNR
        sizes_hf = SNR_hf[peaks_indexes_tp_HF[0], peaks_indexes_tp_HF[1]]
        sizes_lf = SNR_lf[peaks_indexes_tp_LF[0], peaks_indexes_tp_LF[1]]

        # Scale the sizes of the picks
        max_size = 140
        min_size = 2

        # Scale the sizes to the range [0, 1]
        sizes_hf_scaled = min_size + (sizes_hf - np.min(sizes_hf)) / (np.max(sizes_hf) - np.min(sizes_hf)) * (max_size - min_size)
        sizes_lf_scaled = min_size + (sizes_lf - np.min(sizes_lf)) / (np.max(sizes_lf) - np.min(sizes_lf)) * (max_size - min_size)

        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_HF[1] / fs, (peaks_indexes_tp_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='tab:blue', marker='.', s=sizes_hf_scaled, rasterized=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'HF note picks denoised, {metadata["cablename"]}', loc='right')

        # Legend
        # Create a set of legend handles with different sizes
        handles = [
        plt.scatter([], [], s=min(sizes_hf_scaled), color='tab:blue', label=f'Min SNR: {sizes_hf.min():.1f}'),
        plt.scatter([], [], s=(min(sizes_hf_scaled) + max(sizes_hf_scaled)) / 2, color='tab:blue', label=f'Mid SNR: {np.median(sizes_hf):.1f}'),
        plt.scatter([], [], s=max(sizes_hf_scaled), color='tab:blue', label=f'Max SNR: {sizes_hf.max():.1f}')
        ]

        plt.legend(handles=handles, title="SNR Sizes", scatterpoints=1, loc='upper right')
        plt.savefig(f"figs/Figure_{fignum}.pdf")
        fignum += 1
        # plt.show()

        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_LF[1] / fs, (peaks_indexes_tp_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='tab:red', marker='.', s=sizes_lf_scaled, rasterized=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'LF note picks denoised, {metadata["cablename"]}', loc='right')

        # Legend
        # Create a set of legend handles with different sizes
        handles = [
        plt.scatter([], [], s=min(sizes_lf_scaled), color='tab:red', label=f'Min SNR: {sizes_lf.min():.1f}'),
        plt.scatter([], [], s=(min(sizes_lf_scaled) + max(sizes_lf_scaled)) / 2, color='tab:red', label=f'Mid SNR: {np.median(sizes_lf):.1f}'),
        plt.scatter([], [], s=max(sizes_lf_scaled), color='tab:red', label=f'Max SNR: {sizes_lf.max():.1f}')
        ]

        plt.legend(handles=handles, title="SNR Sizes", scatterpoints=1, loc='upper right')
        plt.savefig(f"figs/Figure_{fignum}.pdf")
        fignum += 1
        # plt.show()

        # Make a count of the detections depending on IPI and threshold
        detect_cptHF = [[], []]
        detect_cptLF = [[], []]

        for t in tqdm(thresholds, desc=f"Picking times, iterating over thresholds"):
                peaks_indexes_HF = []
                peaks_indexes_LF = []

                for corr in SNR_hf:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=t)
                        peaks_indexes_HF.append(peaks_indexes)

                for corr in SNR_lf:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = ipi * fs, height=t)  
                        peaks_indexes_LF.append(peaks_indexes)

                peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
                peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

                detect_cptHF[0].append(len(peaks_indexes_tp_HF[0]))
                detect_cptLF[0].append(len(peaks_indexes_tp_LF[0]))
        
        for it in tqdm(IPIs, desc=f"Picking times, iterating over IPIs"):
                peaks_indexes_HF = []
                peaks_indexes_LF = []

                for corr in SNR_hf:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = it * fs, height=th)
                        peaks_indexes_HF.append(peaks_indexes)

                for corr in SNR_lf:
                        peaks_indexes,_ = sp.find_peaks(corr, distance = it * fs, height=th)  
                        peaks_indexes_LF.append(peaks_indexes)

                peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_HF)
                peaks_indexes_tp_LF = dw.detect.convert_pick_times(peaks_indexes_LF)

                detect_cptHF[1].append(len(peaks_indexes_tp_HF[0]))
                detect_cptLF[1].append(len(peaks_indexes_tp_LF[0]))

        # Plot the results of the denoising impact on detection counts
        plt.figure(figsize=(12,10))
        # subplot of the superior line
        plt.subplot(211)
        plt.plot(thresholds, detect_cptHFnoise[0], label='HF noise', lw=2)
        plt.plot(thresholds, detect_cptLFnoise[0], label='LF noise', lw=2)
        plt.plot(thresholds, detect_cptHF[0], label='HF denoised', lw=2, ls='--')
        plt.plot(thresholds, detect_cptLF[0], label='LF denoised', lw=2, ls='--')
        plt.ylabel('Number of detections')
        plt.xlabel('Threshold (dB)')
        plt.legend()
        plt.grid()

        # subplot of the inferior line
        plt.subplot(212)
        plt.plot(IPIs, detect_cptHFnoise[1], label='HF noise', lw=2)
        plt.plot(IPIs, detect_cptLFnoise[1], label='LF noise', lw=2)
        plt.plot(IPIs, detect_cptHF[1], label='HF denoised', lw=2, ls='--')
        plt.plot(IPIs, detect_cptLF[1], label='LF denoised', lw=2, ls='--')
        plt.ylabel('Number of detections')
        plt.xlabel('Inter pulse interval (s)')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
        return      


if __name__ == '__main__':

        # The dataset of this example is constituted of 60s time series along the north and south cables
        url_north = ['http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5']
        
        selected_channels_m_north = [12000, 66000, 5]  # list of values in meters corresponding to the starting,
                                                        # ending and step wanted channels along the FO Cable
                                                        # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                        # in meters

        url_south = [
        'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T015914Z.h5',
        'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T020014Z.h5'
        ]         
        
        selected_channels_m_south = [12000, 95000, 5]

        main(url_north, selected_channels_m_north)
        gc.collect()
        main(url_south, selected_channels_m_south)