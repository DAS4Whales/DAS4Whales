"""
data_handle.py - data handling functions for DAS data

This module provides various functions to handle DAS data, including loading, downloading, and conditioning. 
It aims at having specific functions for each interrogator type.

Authors: Léa Bouffaut, Quentin Goestchel, Erfan Horeh
Date: 2023-2024-2025
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urljoin

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import wget
from nptdms import TdmsFile
from simpledas import simpleDASreader as sd


# Test for the package
def hello_world_das_package() -> None:
    """Print a hello world message for the package."""
    print("Yepee! You now have access to all the functionalities of the das4whale python package!")


# Read metadata
# Definition of the functions for DAS data conditioning
def get_acquisition_parameters(filepath: str, interrogator: str = 'optasense') -> Optional[Dict[str, Any]]:
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

    interrogator_list = ['optasense', 'silixa', 'mars', 'asn', 'onyx']

    if interrogator in interrogator_list:

        if interrogator == 'optasense':
            metadata = get_metadata_optasense(filepath)

        elif interrogator == 'silixa':
            metadata = get_metadata_silixa(filepath)

        elif interrogator == 'mars':
            metadata = get_metadata_mars(filepath)

        elif interrogator == 'asn':
            metadata = get_metadata_asn(filepath)
        
        elif interrogator == 'onyx':
            metadata = get_metadata_onyx(filepath)

    else:
        raise ValueError('Interrogator name incorrect')

    return metadata

def get_metadata_optasense(filepath: str) -> Dict[str, Any]:
    """Gets DAS acquisition parameters for the optasense interrogator e.g., OOI South C1 data

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

def get_metadata_silixa(filepath: str) -> Dict[str, Any]:
    """
    Gets DAS acquisition parameters for the silixa interrogator

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

def get_metadata_asn(filepath: str) -> Dict[str, Any]:
    """
    Gets DAS acquisition parameters for the ASN interrogator e.g., Svalbard data

    Parameters
    ----------
    filepath : string
        a string containing the full path to the data to load

    Returns
    -------
    metadata : dict
        Dictionary filled with metadata, key's breakdown:\n
        fs: the sampling frequency (Hz)\n
        dx: interval between two virtual sensing points also called channel spacing (m)\n
        nx: the number of spatial samples also called channels\n
        ns: the number of time samples\n
        GL: the gauge length (m)\n
        scale_factor: the value to convert DAS data from strain rate to strain

    """

    fp = h5py.File(filepath, 'r')

    fs = 1/fp['header']['dt'][()]  # sampling rate in Hz
    dx = fp['header']['dx'][()]*fp['demodSpec']['roiDec'][()]  # channel spacing in m
    dx = dx[0]
    nx = fp['header']['dimensionRanges']['dimension1']['size'][()] # number of channels
    nx = nx[0]
    ns = fp['header']['dimensionRanges']['dimension0']['size'][()]  # number of samples
    gauge_length = fp['header']['gaugeLength'][()]  # gauge length in m
    n = fp['cableSpec']['refractiveIndex'][()]  # refractive index of the fiber
    scale_factor = fp['header']['sensitivities'][()]
    metadata = {'fs': fs, 'dx': dx, 'ns': ns, 'GL': gauge_length, 'nx': nx, 'scale_factor': scale_factor}
    return metadata

def get_metadata_onyx(filepath: str) -> Dict[str, Any]:
    """Gets DAS acquisition parameters for the onyx interrogator 

    Parameters
    ----------
    filepath : str
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

# Load/download das data as strain
def raw2strain(trace: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Transform the amplitude of raw das data from strain-rate to strain according to scale factor


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
    trace -= np.mean(trace, axis=1, keepdims=True)  # using np.median() is also possible, depending on the nature of the noise
    trace *= metadata["scale_factor"] 
    return trace

def load_das_data(filename: str, selected_channels: List[int], metadata: Dict[str, Any], interrogator: str = 'optasense') -> Tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
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
    interrogator : name of used interrogators. Supports

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

    if interrogator in ['optasense', 'silixa', 'onyx']:

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

    elif interrogator == 'asn':
        dfdas = sd.load_DAS_files(filename, chIndex=None, samples=None, sensitivitySelect=-3,
                                  userSensitivity={'sensitivity': metadata['scale_factor'],
                                                   'sensitivityUnit': 'rad/(m*strain)'},
                                  integrate=True, unwr=True)

        trace = dfdas.values.T
        trace = trace[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)

        # For future save
        file_begin_time_utc = dfdas.meta['time']

    else:
        raise ValueError('Interrogator name incorrect or not supported')


    # Store the following as the dimensions of our data block
    nnx = trace.shape[0]
    nns = trace.shape[1]

    # Define new time and distance axes
    tx = np.arange(nns) / metadata['fs']
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata['dx']

    return trace, tx, dist, file_begin_time_utc

def load_mtpl_das_data(filepaths: List[str], selected_channels: List[int], metadata: Dict[str, Any], timestamp: str, time_window: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
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
    
    # Convert timestamp to microseconds
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


def dl_file(url: str) -> Tuple[str, str]:
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

# Load cable position information
def load_cable_coordinates(filepath: str, dx: float) -> pd.DataFrame:
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


def get_cable_lat_lon_depth(file: str, selected_channels: Tuple[int, int, int]) -> Dict[str, List[float]]:
    """
    Extract latitude, longitude, and depth information from a CSV or TXT file for selected cable channels.

    Parameters
    ----------
    file : str
        The file path to the CSV or TXT file containing latitude, longitude, and depth information.
    selected_channels : tuple
        A tuple containing three integers: (start, stop, step) used to slice and select cable channels.

    Returns
    -------
    position : dict
        A dictionary containing the extracted information for the selected channels:
        - 'chan_idx_oj': list of floats, representing the selected cable channel indices.
        - 'lat': list of floats, representing the selected latitude values.
        - 'lon': list of floats, representing the selected longitude values.
        - 'depth': list of floats, representing the selected depth values in absolute magnitude.
    """
    # Prepare lists
    channel = []
    lat = []
    lon = []
    depth = []

    # Read the latitude, longitude, and depth data from the CSV/TXT file
    with open(file, mode='r') as file:
        csv_file = csv.reader(file, delimiter=' ')
        for lines in csv_file:
            # Filter out empty strings
            lines = [line for line in lines if line]
            channel.append(float(lines[0]))
            lat.append(float(lines[1]))
            lon.append(float(lines[2]))
            depth.append(abs(float(lines[3])))

    # Select the specified channels
    channel = channel[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    lat = lat[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    lon = lon[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    depth = depth[selected_channels[0]:selected_channels[1]:selected_channels[2]]

    # Store latitude, longitude, and depth in a dictionary
    position = {
        'chan_idx_oj': channel,
        'lat': lat,
        'lon': lon,
        'depth': depth,
    }

    return position

# Load annotation files
def load_annotation_csv(filepath: str) -> pd.DataFrame:
    """
    Load the annotation data from a CSV file. The file must include the following columns:
    'file_name', 'apex', 'offset', 'start_time', 'whale_side', as output from the DAS Source Locator
    annotation application (https://github.com/leabouffaut/DASSourceLocator).

    Parameters
    ----------
    filepath : str
        The file path to the annotation CSV file to load.

    Returns
    -------
    annotations : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'file_name': str, names of files
        - 'apex': int, apex values
        - 'offset': int, offset values
        - 'start_time': float, start times
        - 'whale_side': str, indicating the side of the whale (e.g., 'left' or 'right').
    """

    # load the .csv file and create a pandas dataframe
    annotations = pd.read_csv(filepath, header=0, keep_default_na=False)
    annotations.columns = ['file_name', 'apex', 'offset', 'start_time', 'whale_side']

    return annotations


def calc_dist_to_xidx(x: float, selected_channels_m: List[float], selected_channels: List[int], dx: float) -> int:
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


def get_selected_channels(selected_channels_m: List[float], dx: float) -> List[int]:
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


def extract_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename in format YYYY-MM-DDTHHMMSSZ."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6})Z', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%dT%H%M%S').replace(tzinfo=timezone.utc)
    return None

def generate_file_list(base_url: str, start_file: str, duration: int) -> List[str]:
    """
    Generate a list of file URLs that correspond to a given time range, starting from a known file.
    
    Parameters
    ----------
    base_url : str
        The base URL where the files are hosted.
    start_file : str
        Filename of the first file to use as reference.
    duration : int
        Duration in seconds.

    Returns
    -------
    list of str
        List of full file URLs covering the time range.
    """
    # Extract start time from filename
    start_time = extract_timestamp(start_file)
    if start_time is None:
        raise ValueError("Could not extract timestamp from filename.")
    
    selected_files = [urljoin(base_url, start_file)]
    accumulated_time = 0
    current_file = start_file
    
    while accumulated_time < duration:
        # Infer the next file timestamp by checking the difference between current and next
        next_time = start_time + timedelta(seconds=60)  # Default to 60s step
        next_filename = re.sub(r'(\d{4}-\d{2}-\d{2}T\d{6})Z', next_time.strftime('%Y-%m-%dT%H%M%S') + 'Z', current_file)
        
        selected_files.append(urljoin(base_url, next_filename))
        accumulated_time += 60  # Assume 60s duration per file; adjust as needed
        start_time = next_time
        current_file = next_filename
    
    return selected_files