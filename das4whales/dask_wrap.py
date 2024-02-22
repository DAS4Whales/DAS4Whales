# File that wrap up functions in dsp.py in a dask way
import h5py
import wget
import os
import numpy as np
import dask.array as da
from datetime import datetime

import das4whales as dw


def load_das_data(filename, selected_channels, metadata):
    #TODO: Change docstring
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Inputs:
    :param filename: a string containing the full path to the data to load
    :param selected_channels:
    :param metadata: dictionary filled with metadata (sampling frequency, channel spacing, scale factor...)

    Outputs:
    :return: trace: a [channel x sample] nparray containing the strain data
    :return: tx: the corresponding time axis (s)
    :return: dist: the corresponding distance along the FO cable axis (m)
    :return: file_begin_time_utc: the beginning time of the file, can be printed using
    file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S")
    """
    if not os.path.exists(filename):
        raise ValueError('File not found')

    f = h5py.File(filename, 'r') # HDF5 file
    d = f['Acquisition/Raw[0]/RawData']   # Pointer on on-disk array f

    # UTC Time vector for naming
    raw_data_time = f['Acquisition/Raw[0]/RawDataTime']

    # For future save
    file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

    # Store the following as the dimensions of our data block
    nnx = d.shape[0]
    nns = d.shape[1]

    # Define new time and distance axes
    tx = np.arange(nns) / metadata["fs"]
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]
    return d, tx, dist, file_begin_time_utc