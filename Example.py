# File example from Ethan Williams as in
# https://gist.github.com/ethanfwilliams/c7c952220ac329db48f8ef159f0b169f
# downloaded from url = https://www.its.caltech.edu/~efwillia/OOI_RCA_DAS/North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T022302Z.h5

# Below re-download everytime we run the code
# import wget
# url = 'https://www.its.caltech.edu/~efwillia/OOI_RCA_DAS/North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T022302Z.h5'
# das_example_file = wget.download(url)
# print(das_example_file)

import das4whales as dw
import scipy.signal as sp
import numpy as np

# Read HDF5 files and accessing metadata
filename = 'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T022302Z.h5'

# Get the acquisition parameters for the data folder
fs, dx, nx, ns, scale_factor = dw.das_data.get_acquisition_parameters(filename)

# Select desired channels
selected_channels_m = [20000, 65000, 10] # list of values in meters corresponding to the starting,
                                         # ending and step wanted channels along the FO Cable
                                         # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                         # in meters
selected_channels = [int(np.floor(selected_channels_m/dx)) for selected_channels_m in selected_channels_m] # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                         # channels along the FO Cable
                                         # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                         # numbers
# Create conditioning for the signal
# Create high-pass filter
sos_hpfilt = dw.preprocess.butterworth_filter([2,5,'hp'], fs)

# Create band-pass filter for the TX plots
sos_bpfilt = dw.preprocess.butterworth_filter([5,[10, 30],'bp'], fs)

# Load DAS data
tr, tx, dist, fileBeginTimeUTC = dw.das_data.load_DASdata(filename, fs, dx, selected_channels, scale_factor)

# apply the high-pass filter
trf = sp.sosfiltfilt(sos_hpfilt, tr, axis=1)

# FK filter
# loop is taking 1.4s - not much to crunch there
trf_fk = dw.preprocess.fk_filtering(trf, selected_channels, dx, fs, cmin=1450, cmax=3000)

# TX-plot of the FK filtered data, additionally band-pass filtered
trff = sp.sosfiltfilt(sos_bpfilt, trf_fk, axis=1)

# start_time = time.time()
dw.plot.plot_tx(trff, tx, dist, fs, selected_channels, fileBeginTimeUTC)
