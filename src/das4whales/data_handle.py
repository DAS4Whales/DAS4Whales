"""
data_handle.py - data handling functions for DAS data

This module provides various functions to handle DAS data, including loading, downloading, and conditioning. 
It aims at having specific functions for each interrogator type.

Authors: Léa Bouffaut, Quentin Goestchel, Erfan Horeh
Date: 2023-2024
"""

import h5py
import wget
import os
import numpy as np
import dask.array as da
from datetime import datetime, timezone
import pandas as pd
from nptdms import TdmsFile


def hello_world_das_package():
    print("Yepee! You now have access to all the functionalities of the das4whale python package!")


# Definition of the functions for DAS data conditioning
def get_acquisition_parameters(filepath, interrogator='optasense'):
    """
    Retrieve acquisition parameters based on the specified interrogator.

    Parameters
    ----------
    filepath : str
        The file path to the data file.
    interrogator : str, optional
        The interrogator type, one of {'optasense', 'silixa', 'mars', 'alcatel', 'onyx'}.
        Defaults to 'optasense'.

    Returns
    -------
    metadata : dict or None
        Metadata related to the acquisition parameters. Returns None if no matching
        interrogator is found.

    Raises
    ------
    ValueError
        If the interrogator name is not in the predefined list.
    """
    # List the known used interrogators:
    interrogator_list = ['optasense', 'silixa', 'mars', 'alcatel', 'onyx']
    if interrogator in interrogator_list:

        if interrogator == 'optasense':
            metadata = get_metadata_optasense(filepath)

        elif interrogator == 'silixa':
            metadata = get_metadata_silixa(filepath)

        elif interrogator == 'mars':
            metadata = get_metadata_mars(filepath)

        elif interrogator == 'alcatel':
            metadata = get_metadata_alcatel(filepath)
        
        elif interrogator == 'onyx':
            metadata = get_metadata_onyx(filepath)

    else:
        raise ValueError('Interrogator name incorrect')

    return metadata


def get_metadata_optasense(filepath):
    """Gets DAS acquisition parameters for the optasense interrogator 

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        n: refractive index of the fiber\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """
    # Make sure the file exists
    if os.path.exists(filepath):
        # Ensure the closure of the file after reading
        with h5py.File(filepath, 'r') as fp1:
            fp1 = h5py.File(filepath, 'r')

            fs = fp1['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
            dx = fp1['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
            ns = fp1['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']
            n = fp1['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
            GL = fp1['Acquisition'].attrs['GaugeLength'] # gauge length in m
            nx = fp1['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
            scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * GL)

        meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    else:
        raise FileNotFoundError(f'File {filepath} not found')

    return meta_data


def get_metadata_silixa(filepath):
    """Gets DAS acquisition parameters for the silixa interrogator 

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        n: refractive index of the fiber\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """

    # Make sure the file exists
    if os.path.exists(filepath):
        fp = TdmsFile.read(filepath)
        props = fp.properties
        group = fp['Measurement']
        acousticData = np.asarray( [group[channel].data for channel in group] )

        fs = props['SamplingFrequency[Hz]'] # sampling rate in Hz
        dx = props['SpatialResolution[m]'] # channel spacing in m
        ns = acousticData.shape[1]
        n =  props['FibreIndex'] # refractive index
        GL =  props['GaugeLength'] # gauge length in m
        nx = acousticData.shape[0] # number of channels
        scale_factor = (116 * fs * 10**-9) / (GL * 2**13)

        meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    else:
        raise FileNotFoundError(f'File {filepath} not found')
    
    return meta_data


def get_metadata_onyx(filepath):
    """Gets DAS acquisition parameters for the onyx interrogator 

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        n: refractive index of the fiber\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """

    # Make sure the file exists
    if os.path.exists(filepath):
        fp1 = h5py.File(filepath, 'r')
        fs = fp1['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
        dx = fp1['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
        ns = fp1['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']
        n = 1.4682 # refractive index TODO: check if it is correct, it is not in the metadata
        GL = fp1['Acquisition'].attrs['GaugeLength'] # gauge length in m
        nx = fp1['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
        scale_factor = 115e-9 # According to Brad
        print(scale_factor)

        meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    else:
        raise FileNotFoundError(f'File {filepath} not found')
    
    return meta_data


def raw2strain(trace, metadata):
    """Transform the amplitude of raw das data from strain-rate to strain according to scale factor


    Parameters
    ----------
    trace : array-like
        a [channel x time sample] nparray containing the raw data in the spatio-temporal domain
    metadata : dict
        dictionary filled with metadata (fs, dx, nx, ns, n, GL, scale_factor)

    Returns
    -------
    trace : array-like
        a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    """    
    # Remove the mean trend from each channel and scale

    trace -= np.mean(trace, axis=1, keepdims=True)
    trace *= metadata["scale_factor"] 
    return trace


def load_das_data(filename, selected_channels, metadata):
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Parameters
    ----------
    filename : str
        A string containing the full path to the data to load.
    selected_channels : list
        A list containing the selected channels.
    metadata : dict
        A dictionary filled with metadata (sampling frequency, channel spacing, scale factor...).

    Returns
    -------
    trace : np.ndarray
        A [channel x sample] nparray containing the strain data.
    tx : np.ndarray
        The corresponding time axis (s).
    dist : np.ndarray
        The corresponding distance along the FO cable axis (m).
    file_begin_time_utc : datetime.datetime
        The beginning time of the file, can be printed using file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S").
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File {filename} not found')

    with h5py.File(filename, 'r') as fp:
        # Data matrix
        raw_data = fp['Acquisition/Raw[0]/RawData']

        # Check the orientation of the data compared to the metadata
        if raw_data.shape[0] == metadata["nx"]:
            # Data is in the correct orientation
            pass
        elif raw_data.shape[1] == metadata["nx"]:
            # Data is transposed without loading in memory
            raw_data = raw_data[:,:].T

        # Selection the traces corresponding to the desired channels
        # Loaded as float64, float 32 might be sufficient?
        trace = raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)
        trace = raw2strain(trace, metadata)

        # UTC Time vector for naming
        raw_data_time = fp['Acquisition']['Raw[0]']['RawDataTime']

        # For future save
        file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)

        # Store the following as the dimensions of our data block
        nnx = trace.shape[0]
        nns = trace.shape[1]

        # Define new time and distance axes
        tx = np.arange(nns) / metadata["fs"]
        dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]

    return trace, tx, dist, file_begin_time_utc


def load_mtpl_das_data(filepaths, selected_channels, metadata, timestamp, time_window):
    """
    Load the DAS data corresponding to the input file names as strain according to the selected channels. Takes multiple files as input and concatenates them along the time axis starting from the input timestamp for the input time window.

    Parameters
    ----------
    filepaths : list
        A list containing the full pathes to the data to load.
    selected_channels : list
        A list containing the selected channels.
    metadata : dict
        A dictionary filled with metadata (sampling frequency, channel spacing, scale factor...).
    timestamp : str
        The timestamp to extract the data from.
    time_window : float
        The time window duration to extract the data from.

    Returns
    -------
    trace : np.ndarray
        A [channel x sample] nparray containing the strain data.
    tx : np.ndarray
        The corresponding time axis (s).
    dist : np.ndarray
        The corresponding distance along the FO cable axis (m).
    file_begin_time_utc : datetime.datetime
        The beginning time of the file, can be printed using file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S").
    """

    # Print the input timestamp
    print(f'timestamp_input: {timestamp}')
    
    # Convert timestamp to microseconds since epoch
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp_us = timestamp.timestamp() * 1e6

    trace = None
    file_begin_time_utc = None
    
    # Loop through each filepath lazily
    for filepath in filepaths:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File {filepath} not found')
        
        with h5py.File(filepath, 'r') as fp:
            raw_data = fp['Acquisition/Raw[0]/RawData']
            raw_data_time = fp['Acquisition/Raw[0]/RawDataTime']

            # Select the traces corresponding to the desired channels lazily
            selected_trace = da.from_array(raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :], chunks='auto')
            
            # Find the index where raw_data_time >= timestamp lazily
            if trace is None:
                index = np.searchsorted(raw_data_time, timestamp_us)
                file_begin_time_utc = datetime.fromtimestamp(raw_data_time[index] * 1e-6, tz=timezone.utc)
            
            # Concatenate traces along the time axis lazily
            trace = selected_trace if trace is None else da.concatenate([trace, selected_trace], axis=1)

    # Convert the time window duration to samples
    duration_samples = int(time_window * metadata["fs"])
    
    # Extract the desired time window lazily
    trace = trace[:, index:index + duration_samples].astype(np.float64)

    # Convert raw data to strain
    tr = raw2strain(trace, metadata)

    print(f'timestamp_output: {file_begin_time_utc}')

    # Get dimensions of the data block lazily
    nnx, nns = trace.shape
    
    # Define new time and distance axes lazily
    time = np.arange(nns) / metadata["fs"]
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]

    # Return dask arrays for lazy evaluation
    return tr.compute(), time, dist, file_begin_time_utc


def dl_file(url):
    """Download the file at the given url

    Parameters
    ----------
    url : string
        url location of the file

    Returns
    -------
    filepath : string
        local path destination of the file
    """    
    filename = url.split('/')[-1]
    filepath = os.path.join('data',filename)
    if os.path.exists(filepath) == True:
        print(f'{filename} already stored locally')
    else:
        # Create the data subfolder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        wget.download(url, out='data', bar=wget.bar_adaptive)
        print(f'Downloaded {filename}')
    return filepath, filename


def load_cable_coordinates(filepath, dx):
    """
    Load the cable coordinates from a text file.

    Parameters
    ----------
    filepath : str
        The file path to the cable coordinates file.
    dx : float
        The distance between two channels.

    Returns
    -------
    df : pandas.DataFrame
        The cable coordinates dataframe.
    """

    # load the .txt file and create a pandas dataframe
    df = pd.read_csv(filepath, delimiter = ",", header = None)
    df.columns = ['chan_idx','lat', 'lon', 'depth']
    df['chan_m'] = df['chan_idx'] * dx

    return df


def calc_dist_to_xidx(x, selected_channels_m, selected_channels, dx):
    """
    Calculate the index of the channel closest to the given distance.

    Parameters
    ----------
    x : float
        The distance along the cable.
    selected_channels_m : list
        The selected channels in meters.
    selected_channels : list
        The selected channels.
    dx : float
        The distance between two channels.

    Returns
    -------
    int
        The index of the channel closest to the given distance.
    """
    return int((x-selected_channels_m[0]) / (dx * selected_channels[2]))


def get_selected_channels(selected_channels_m, dx):
    """
    Get the selected channels in channel numbers.

    Parameters
    ----------
    selected_channels_m : list
        The selected channels in meters. [ChannelStart_m, ChannelStop_m, ChannelStep_m]
    dx : float
        The distance between two channels.

    Returns
    -------
    list
        The selected channels in channel numbers. [ChannelStart, ChannelStop, ChannelStep]
    """
    selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                     selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                           # channels along the FO Cable
                                           # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                           # numbers
    return selected_channels