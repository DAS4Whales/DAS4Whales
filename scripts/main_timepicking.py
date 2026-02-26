
# Libraries import
import numpy as np
import xarray as xr
import scipy.signal as sp
import matplotlib.pyplot as plt
import das4whales as dw
import cv2
import gc
from tqdm import tqdm
import argparse
import matplotlib.cm as cm
import matplotlib.colors as colors

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

                metadata["selected_channels"] = selected_channels
                metadata["selected_channels_m"] = selected_channels_m
                ### Load raw DAS data
                
                # Loads the data using the pre-defined selected channels. 

                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
                metadata["fileBeginTimeUTC"] = fileBeginTimeUTC.strftime("%Y-%m-%d_%H:%M:%S")
                metadata["data_shape"] = tr.shape
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

                metadata["selected_channels"] = selected_channels
                metadata["selected_channels_m"] = selected_channels_m

                timestamp = '2021-11-04 02:00:02.025000'
                duration = 60
                selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, dx)

                # Load the data
                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_mtpl_das_data(filepaths, selected_channels, metadata, timestamp, duration)
                metadata["fileBeginTimeUTC"] = fileBeginTimeUTC.strftime("%Y-%m-%d_%H:%M:%S")
                metadata["data_shape"] = tr.shape

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
        
        ########  Perform the time picking on the denoised mf traces #########
        # Inter pulse interval and threshold
        ipi = 2
        th = 4
        fignum = 1

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

        flat_snr_hf = SNR_hf[peaks_indexes_tp_HF[0], peaks_indexes_tp_HF[1]]
        flat_snr_lf = SNR_lf[peaks_indexes_tp_LF[0], peaks_indexes_tp_LF[1]]

        # Save the time picking results, along with the metadata
        ds = xr.Dataset(
                {
                        'peaks_indexes_tp_HF': (['coord', 'peak_HF'], peaks_indexes_tp_HF),
                        'peaks_indexes_tp_LF': (['coord', 'peak_LF'], peaks_indexes_tp_LF),
                        'SNR_hf': (['peak_HF'], flat_snr_hf),  # 
                        'SNR_lf': (['peak_LF'], flat_snr_lf),  # 
                },
                coords={
                        "coord": ["time_idx", "dist_idx"],  # Naming the tuple components
                        "peak_HF": range(flat_snr_hf.shape[0]),  # Define coordinate for HF peaks
                        "peak_LF": range(flat_snr_lf.shape[0]),  # Define coordinate for LF peaks
                },
                attrs=metadata
        )

        ds.to_netcdf(f'out/peaks_indexes_tp_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}_ipi{ipi}_th_{th}.nc')

        # # Save the SNR matrixes 
        # np.save(f'out/SNR_hf_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}.npy', SNR_hf)
        # np.save(f'out/SNR_lf_{metadata["cablename"]}_{metadata["fileBeginTimeUTC"]}.npy', SNR_lf)

        print('HF detections after denoising:', len(peaks_indexes_tp_HF[0]))
        print('LF detections after denoising:', len(peaks_indexes_tp_LF[0]))

        # Determine common color scale
        vmin = min(np.min(flat_snr_hf), np.min(flat_snr_lf))
        vmax = max(np.max(flat_snr_hf), np.max(flat_snr_lf))
        cmap = cm.plasma  # Define colormap
        norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Normalize color range

        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_HF[1] / fs, (peaks_indexes_tp_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, 
                         c=flat_snr_hf, cmap=cmap, norm=norm, s=flat_snr_hf)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'HF note picks denoised, {metadata["cablename"]}', loc='right')
        plt.grid(linestyle='--', alpha=0.5)
        plt.colorbar(label='SNR', orientation='vertical', fraction=0.02, pad=0.02)
        plt.savefig(f"figs/tpicks/Peaks_HF_{metadata['cablename']}_{metadata['fileBeginTimeUTC']}_ipi{ipi}_th_{th}.pdf")
        # plt.show()

        plt.figure(figsize=(12,10))
        plt.scatter(peaks_indexes_tp_LF[1] / fs, (peaks_indexes_tp_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, 
                         c=flat_snr_lf, cmap=cmap, norm=norm, s=flat_snr_lf)
        plt.grid(linestyle='--', alpha=0.5)
        plt.colorbar(label='SNR', orientation='vertical', fraction=0.02, pad=0.02)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'LF note picks denoised, {metadata["cablename"]}', loc='right')
        plt.savefig(f"figs/tpicks/Peaks_LF_{metadata['cablename']}_{metadata['fileBeginTimeUTC']}_ipi{ipi}_th_{th}.pdf")
        # plt.tight_layout()
        # plt.show()
        return


def parse_input_line(input_line):
    """
    Parses a single input line containing timestamp, file lists, and channel parameters.

    Expected format of the input line:
        TIMESTAMP NorthFILE1 NorthFILE2 ... northCHANNEL_MIN northCHANNEL_MAX northCHANNEL_STEP \
                  SouthFILE1 SouthFILE2 ... southCHANNEL_MIN southCHANNEL_MAX southCHANNEL_STEP

    Example:
        2025-03-29T12:00:00 north_file1.h5 north_file2.h5 12000 66000 5 
                             south_file1.h5 south_file2.h5 12000 95000 5

    - TIMESTAMP: UTC timestamp of the window (YYYY-MM-DDTHH:MM:SS)
    - NorthFILEs: List of North cable file URLs (can be multiple)
    - northCHANNEL_MIN, northCHANNEL_MAX, northCHANNEL_STEP: Integer values for channel range
    - SouthFILEs: List of South cable file URLs (can be multiple)
    - southCHANNEL_MIN, southCHANNEL_MAX, southCHANNEL_STEP: Integer values for channel range

    Parameters
    ----------
    input_line : str
        A space-separated string containing all the required information.

    Returns
    -------
    tuple
        (timestamp, list of north files, list of north channel params, list of south files, list of south channel params)
    """
    # Split the input line into components
    input_line = input_line.split()
    
    # Extract timestamp (first element)
    timestamp_str = input_line[0]
    timestamp = timestamp_str.replace("T", " ")  # Replace 'T' with space for datetime

    # Initialize lists for north and south files
    north_files = []
    i = 1
    # Collect all north files (assuming they start with "http")
    while input_line[i].startswith("http"):
        north_files.append(input_line[i])
        i += 1
    
    # Extract north channel parameters (next three elements)
    selected_channels_m_north = [int(input_line[i]), int(input_line[i+1]), int(input_line[i+2])]
    i += 3  # Move past the channel parameters

    # Initialize south files list
    south_files = []
    # Collect all south files (assuming they start with "http")
    while input_line[i].startswith("http"):
        south_files.append(input_line[i])
        i += 1
    
    # Extract south channel parameters (next three elements)
    selected_channels_m_south = [int(input_line[i]), int(input_line[i+1]), int(input_line[i+2])]

    # Return the parsed components as a tuple
    return timestamp, north_files, selected_channels_m_north, south_files, selected_channels_m_south


if __name__ == '__main__':
        # parser = argparse.ArgumentParser(description="Process DAS data files.")
        # parser.add_argument("--input", type=str, required=True, help="Full input line")
        # args = parser.parse_args()

        # # Parse the input line
        # timestamp, urls_north, selected_channels_m_north, urls_south, selected_channels_m_south = parse_input_line(args.input)

        # print(f"Timestamp: {timestamp}")
        # print(f"North files: {urls_north}")
        # print(f"North channel params: {selected_channels_m_north}")
        # print(f"South files: {urls_south}")
        # print(f"South channel params: {selected_channels_m_south}")

        # main(urls_north, selected_channels_m_north, timestamp)
        # # gc.collect()
        # main(urls_south, selected_channels_m_south, timestamp)

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