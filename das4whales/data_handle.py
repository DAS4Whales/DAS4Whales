import h5py
import numpy as np
from datetime import datetime


def hello_world_das_data():
    print("Yepee! You now have access to all the functionalities of the das4whale python package!")


# Definition of the functions for DAS data conditioning
def get_acquisition_parameters(filename):
    """
    Gets DAS acquisition parameters

    Inputs:
    - filename, a string containing the full path to the data to load

    Outputs:
    - fs, the sampling frequency (Hz)
    - dx, interval between two virtual sensing points also called channel spacing (m)
    - nx, the number of spatial samples also called channels
    - ns, the number of time samples
    - gauge_length, the gauge length (m)
    - scale_factor, the value to convert DAS data from strain rate to strain

    """

    fp = h5py.File(filename, 'r')

    fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']  # sampling rate in Hz
    dx = fp['Acquisition'].attrs['SpatialSamplingInterval']  # channel spacing in m
    nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']  # number of channels
    ns = fp['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']  # number of samples
    gauge_length = fp['Acquisition'].attrs['GaugeLength']  # gauge length in m
    n = fp['Acquisition']['Custom'].attrs['Fibre Refractive Index']  # refractive index of the fiber
    scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * gauge_length)

    return fs, dx, nx, ns, gauge_length, scale_factor


def raw2strain(trace, scale_factor):
    """
    Transform the amplitude of raw das data from strain-rate to strain according to scale factor

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
    - trace, a [channel x sample] nparray containing the strain data
    - tx, the corresponding time axis (s)
    - dist, the corresponding distance along the FO cable axis (m)
    - file_begin_time_utc, the beginning time of the file, can be printed using file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S")

    """

    with h5py.File(filename, 'r') as fp:
        # Data matrix
        raw_data = fp['Acquisition']['Raw[0]']['RawData']

        # Selection the traces corresponding to the desired channels
        trace = raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :]
        trace = raw2strain(trace, scale_factor)

        # UTC Time vector for naming
        raw_data_time = fp['Acquisition']['Raw[0]']['RawDataTime']

        # For future save
        file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

        # Store the following as the dimensions of our data block
        nnx = trace.shape[0]
        nns = trace.shape[1]

        # Define new time and distance axes
        tx = np.arange(nns) / fs
        dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * dx

    return trace, tx, dist, file_begin_time_utc
