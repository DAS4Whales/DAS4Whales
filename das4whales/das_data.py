import h5py
import numpy as np
from datetime import datetime


def hello_world_das_data():
    print("You have access to the das data read and load functions")


# Definition of the functions for DAS data conditioning
def get_acquisition_parameters(filename):
    """
    Gets DAS acquisition parameters

    Inputs:
    - filename, a string containing the full path to the data to load

    Outputs:
    - fs, the sampling frequency (Hz)
    - dx, the channel spacing (m)
    - nx, the number of channels
    - ns, the number of samples
    - scale_factor, the value to convert DAS data into strain

    """

    # From the first file in the folder, get all the information we will further need
    fp = h5py.File(filename, 'r')

    fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']  # sampling rate in Hz
    dx = fp['Acquisition'].attrs['SpatialSamplingInterval']  # channel spacing in m
    nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']  # number of channels
    ns = fp['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']  # number of samples
    GL = fp['Acquisition'].attrs['GaugeLength']  # gauge length in m
    n = fp['Acquisition']['Custom'].attrs['Fibre Refractive Index']  # refractive index
    scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * GL)

    return fs, dx, nx, ns, scale_factor


def raw2strain(trace, scale_factor):
    """
    Transform raw data into strain

    Inputs:
    - trace, a channel x sample nparray containing the raw recorded data data

    Outputs:
    - tr, a channel x sample nparray containing the strain data

    """
    # Remove the mean trend from each channel and scale
    mn = np.tile(np.mean(trace, axis=1), (trace.shape[1], 1)).T
    trace = trace - mn
    trace *= scale_factor
    return trace


def load_das_data(filename, fs, dx, selected_channels, scale_factor):
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Inputs:
    - filename, a string containing the full path to the data to load

    Outputs:
    - trace, a channel x sample nparray containing the strain data
    - tx, the corresponding time axis (s)
    - dist, the corresponding distance along the FO cable axis (m)
    - file_begin_time_utc, the beginning time of the file, can be printed using file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S")

    """

    with h5py.File(filename, 'r') as fp:
        # Data matrix
        raw_data = fp['Acquisition']['Raw[0]']['RawData']

        # Selection the traces corresponding tot he desired channels
        trace = raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :]  # .astype(float)
        trace = raw2strain(trace, scale_factor)

        # UTC Time vector for naming
        raw_data_time = fp['Acquisition']['Raw[0]']['RawDataTime']
        # rawDataTimeArr = raw_data_time[:]

        # For future save
        file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

        # Store the following as the dimensions of our data block
        nnx = trace.shape[0]
        nns = trace.shape[1]

        # Define new time and distance axes
        tx = np.arange(nns) / fs
        dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * dx

    return trace, tx, dist, file_begin_time_utc
