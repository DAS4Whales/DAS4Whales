"""
data_handle.py - data handling functions for DAS data

This module provides various functions to handle DAS data, including loading, downloading, and conditioning. 
It aims at having specific functions for each interrogator type.

Authors: Léa Bouffaut, Quentin Goestchel, Erfan Horeh
Date: 2023-2024-2025
"""

import h5py
import wget
import os
import numpy as np
import csv
import dask.array as da
from datetime import datetime, timezone
from simpledas import simpleDASreader as sd
import pandas as pd
from nptdms import TdmsFile

# Test for the package
def hello_world_das_package():
    print("Yepee! You now have access to all the functionalities of the das4whale python package!")


# Read metadata
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

def get_metadata_optasense(filepath):
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

def get_metadata_silixa(filepath):
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

def get_metadata_asn(filepath):
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

# Load/download das data as strain
def raw2strain(trace, metadata):
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

def load_das_data(filename, selected_channels, metadata, interrogator='optasense'):
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

def load_das_file_startTime(filename, interrogator='optasense'):
    """loads just the start time of a file
    returns:
        file_begin_time_utc : datetime.datetime"""
    if interrogator in ['optasense', 'silixa', 'onyx']:
        # UTC Time vector for naming
        raw_data_time = fp['Acquisition']['Raw[0]']['RawDataTime']

        # For future save
        file_begin_time_utc = datetime.utcfromtimestamp(raw_data_time[0] * 1e-6)
    elif interrogator == 'asn':
        print('WARNING: This is likely VERY slow!!!!')
        # TODO: try to use get_filemeta(filepath: str, metaDetail: int = 1):
        dfdas = sd.load_DAS_files(filename, chIndex=0, samples=0, sensitivitySelect=-3,
                                  userSensitivity={'sensitivity': metadata['scale_factor'],
                                                   'sensitivityUnit': 'rad/(m*strain)'},
                                  integrate=True, unwr=True)
        file_begin_time_utc = dfdas.meta['time']
    else:
        raise ValueError('Interrogator Name incorrect or not supported.')
    return file_begin_time_utc

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

# Load cable position information
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


def get_cable_lat_lon_depth(file, selected_channels):
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
def load_annotation_csv(filepath):
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

class iterative_loader:
    """
    Class for loading DAS directories in chunks
    """
    def __init__(self, dirpath, selected_channels, metadata=None, interrogator='optasense', 
                 start_file_index=0, end_file_index=None, time_window_s=30):
        """
        Initialize the iterative_loader class.

        Parameters
        ----------
        dirpath : str
            The directory path to the DAS data files.
        selected_channels : list
            A list containing the selected channels.
        metadata : dict
            A dictionary filled with metadata (sampling frequency, channel spacing, scale factor...).
        interrogator : str, optional
            The interrogator type, one of {'optasense', 'silixa', 'mars', 'alcatel', 'onyx'}.
            Defaults to 'optasense'.
        start_file_index : int, optional
            Index of first file to process. Defaults to 0.
        end_file_index : int, optional
            Index of last file to process. Defaults to None (process all files).
        time_window_s : float, optional
            Time window in seconds for each chunk. Defaults to 30.
        """
        self.dirpath = dirpath
        self.selected_channels = selected_channels
        
        if metadata is None:
            metadata_loader = MetadataLoader(dirpath)
            metadata = metadata_loader.load_metadata()
        self.metadata = metadata
        self.interrogator = interrogator
        
        # Get and sort file list
        self.file_list = [os.path.join(dirpath, f) for f in os.listdir(dirpath) 
                         if f.endswith(('.h5', '.hdf5', '.tdms'))]
        self.file_list.sort()
        
        self.current_file_index = start_file_index
        self.start_file_index = start_file_index
        self.end_file_index = end_file_index if end_file_index is not None else len(self.file_list)
        self.time_window_s = time_window_s
        
        # Validate file indices
        if self.start_file_index >= len(self.file_list):
            raise ValueError(f"start_file_index ({start_file_index}) exceeds number of files ({len(self.file_list)})")
        if self.end_file_index > len(self.file_list):
            raise ValueError(f"end_file_index ({end_file_index}) exceeds number of files ({len(self.file_list)})")
        
        self.data_in_memory = {}
        self._load_initial_data()

    def _load_initial_data(self):
        """Load initial data and prepare first chunk"""
        if self.current_file_index >= self.end_file_index:
            raise ValueError("No files to process in the specified range")
            
        # Load first file
        trace, tx, dist, file_begin_time_utc = load_das_data(
            self.file_list[self.current_file_index], 
            self.selected_channels, 
            self.metadata, 
            self.interrogator
        )
        
        self.nx, self.ns = trace.shape
        self.data_in_memory = {
            'trace': trace,
            'tx': tx,
            'dist': dist,
            'file_begin_time_utc': file_begin_time_utc
        }
        
        # Keep track of absolute time offset from the very first file
        self.absolute_time_offset = 0.0
        
        # Load additional files if needed to fill time window
        self._ensure_sufficient_data()

    def _ensure_sufficient_data(self):
        """Ensure we have enough data loaded to fill the current time window"""
        while (len(self.data_in_memory['tx']) == 0 or 
               self.data_in_memory['tx'][-1] < self.time_window_s):
            
            self.current_file_index += 1
            if self.current_file_index >= self.end_file_index:
                break
                
            try:
                trace_next, tx_next, dist_next, file_begin_time_utc_next = load_das_data(
                    self.file_list[self.current_file_index], 
                    self.selected_channels, 
                    self.metadata, 
                    self.interrogator
                )
                
                # Calculate time offset for next file
                if len(self.data_in_memory['tx']) > 0:
                    time_offset = self.data_in_memory['tx'][-1] + 1/self.metadata['fs']
                    tx_next_adjusted = tx_next + time_offset
                else:
                    tx_next_adjusted = tx_next
                
                # Concatenate data
                self.data_in_memory['trace'] = np.concatenate(
                    (self.data_in_memory['trace'], trace_next), axis=1
                )
                self.data_in_memory['tx'] = np.concatenate(
                    (self.data_in_memory['tx'], tx_next_adjusted)
                )
                
            except Exception as e:
                print(f'Error loading file {self.file_list[self.current_file_index]}: {e}, skipping...')
                continue

    def get_next_chunk(self, new_time_window_s=None):
        """
        Get the next chunk of DAS data.

        Parameters
        ----------
        new_time_window_s : float, optional
            New time window size. If provided, updates the time window for subsequent calls.

        Returns
        -------
        trace : np.ndarray
            A [channel x sample] array containing the strain data.
        tx : np.ndarray
            The corresponding time axis (s) relative to chunk start.
        dist : np.ndarray
            The corresponding distance along the FO cable axis (m).
        section_begin_time_utc : datetime.datetime
            The beginning time of this chunk.
        """
        if new_time_window_s is not None:
            self.time_window_s = new_time_window_s

        # Check if we have any data left
        if len(self.data_in_memory['tx']) == 0:
            self._ensure_sufficient_data()
            
        if len(self.data_in_memory['tx']) == 0:
            raise StopIteration("No more data available.")

        # Ensure we have enough data for the time window
        self._ensure_sufficient_data()
        
        # Determine how much data to extract
        if len(self.data_in_memory['tx']) == 0:
            raise StopIteration("No more data available.")
            
        # Find indices for the current time window
        available_time = self.data_in_memory['tx'][-1] if len(self.data_in_memory['tx']) > 0 else 0
        actual_window = min(self.time_window_s, available_time)
        
        idx = self.data_in_memory['tx'] <= actual_window
        
        if not np.any(idx):
            raise StopIteration("No more data available.")
        
        # Extract data for this chunk
        trace = self.data_in_memory['trace'][:, idx].copy()
        tx_chunk = self.data_in_memory['tx'][idx].copy()
        dist = self.data_in_memory['dist'].copy()
        
        # Calculate section begin time
        section_begin_time_utc = (self.data_in_memory['file_begin_time_utc'] + 
                                 pd.to_timedelta(self.absolute_time_offset, unit='s'))
        
        # Remove extracted data from memory
        self.data_in_memory['trace'] = self.data_in_memory['trace'][:, ~idx]
        self.data_in_memory['tx'] = self.data_in_memory['tx'][~idx] - actual_window
        
        # Update absolute time offset
        self.absolute_time_offset += actual_window
        
        # Reset tx_chunk to start from 0
        tx_chunk = tx_chunk - tx_chunk[0] if len(tx_chunk) > 0 else tx_chunk
        
        return trace, tx_chunk, dist, section_begin_time_utc

    def __iter__(self):
        """Make the class iterable"""
        return self
    
    def __next__(self):
        """Iterator protocol"""
        return self.get_next_chunk()
    
    def reset(self):
        """Reset the loader to the beginning"""
        self.current_file_index = self.start_file_index
        self.absolute_time_offset = 0.0
        self._load_initial_data()