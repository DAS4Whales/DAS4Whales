# Download directly from the OOI DAS experiment - details here:
# https://oceanobservatories.org/pi-instrument/ \
# rapid-a-community-test-of-distributed-acoustic-sensing-on-the-ocean-observatories-initiative-regional-cabled-array/

import wget
import os
import das4whales as dw
import scipy.signal as sp
import numpy as np

filename = 'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

# Check if the file exists otherwise download it
# Files are ~850 MB so the download can take a while

if os.path.exists(filename):
    print(filename, ' already exists in path')
else:
    url = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/' \
          'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/' \
          'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T022302Z.h5'

    das_example_file = wget.download(url)
    print(['Downloaded: ', das_example_file])

# Read HDF5 files and accessing metadata
# Get the acquisition parameters for the data folder
fs, dx, nx, ns, gauge_length, scale_factor = dw.data_handle.get_acquisition_parameters(filename)

# Select desired channels
selected_channels_m = [20000, 65000,
                       10]  # [20000, 65000, 10]  # list of values in meters corresponding to the starting,
# ending and step wanted channels along the FO Cable
# selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
# in meters
selected_channels = [int(np.floor(selected_channels_m / dx)) for selected_channels_m in
                     selected_channels_m]  # list of values in channel number (spatial sample) corresponding
# to the starting, ending and step wanted
# channels along the FO Cable
# selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
# numbers
# Create conditioning for the signal

# Create high-pass filter
sos_hpfilt = dw.dsp.butterworth_filter([2, 5, 'hp'], fs)

# Create band-pass filter for the TX plots
sos_bpfilt = dw.dsp.butterworth_filter([5, [10, 30], 'bp'], fs)

# Load DAS data
tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filename, fs, dx, selected_channels, scale_factor)

# apply the high-pass filter
trf = sp.sosfiltfilt(sos_hpfilt, tr, axis=1)

# FK filter
# Create the f-k filter
fk_filter = dw.dsp.fk_filter_design((trf.shape[0],trf.shape[1]), selected_channels, dx, fs, cs_min=1400, cp_min=1480, cp_max=3400, cs_max=3500)

# Apply the f-k filter to the data
trf_fk = dw.dsp.fk_filter_filt(trf,fk_filter)

# Plot
dw.plot.plot_tx(trf_fk, time, dist, fileBeginTimeUTC, fig_size=(12, 10))

# Spatio-temporal plot
dw.plot.plot_tx(trf_fk, time, dist, fileBeginTimeUTC)

# Spatio-spectral plot
# dw.plot.plot_fx(trff, dist, fs, win_s=5,  nfft=512, f_min=0, f_max=50)

# Make audio
