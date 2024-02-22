import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal, ndimage
import dask

def fk_filt_chunk(data,tint,fs,xint,dx,c_min,c_max):
    '''
    fk_filt_chunk - perform fk filtering on single chunk of DAS data

    Parameters
    ----------
    data : xr.DataArray
        DataArray containing single chunk
    tint : float
        interval in time between samples
    fs : float
        sampling frequency
    xint : float
        interval in space between samples
    dx : float
        distance between samples

    '''

    data_fft = np.fft.fft2(signal.detrend(data))
    
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = np.fft.fftshift(np.fft.fftfreq(ns, d = tint/fs))
    k = np.fft.fftshift(np.fft.fftfreq(nx, d = xint*dx))
    ff,kk = np.meshgrid(f,k)

    # Soundwaves have f/k = c so f = k*c

    g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
    g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

    g = g + np.fliplr(g)
    g2 = g2 + np.fliplr(g2)
    g = g-g2
    g = ndimage.gaussian_filter(g, 40)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    g = (g - np.min(g.flatten())) / (np.max(g.flatten()) - np.min(g.flatten()))
    g = g.astype('f')

    data_fft_g = np.fft.fftshift(data_fft) * g
    data_g = np.fft.ifft2(np.fft.ifftshift(data_fft_g))
    
    #return f,k,g,data_fft_g,data_g
    
    # construct new DataArray
    data_gx = xr.DataArray(data_g, dims=['distance','time'], coords=data.coords)
    return data_gx


def fk_filt(data,tint,fs,xint,dx,c_min,c_max):
    '''
    fk_filt - perform fk filtering on DAS data

    Parameters
    ----------
    data : xr.DataArray
        DataArray containing DAS data
    tint : float
        interval in time between samples
    fs : float
        sampling frequency
    xint : float
        interval in space between samples
    dx : float
        distance between samples

    '''
    kwargs = {'tint':tint, 'fs':fs, 'xint':xint, 'dx':dx, 'c_min':c_min, 'c_max':c_max}
    data_gx = data.map_blocks(fk_filt_chunk, kwargs=kwargs, template=data)
    return data_gx


def _energy_TimeDomain_chunk(da, time_dim='time'):
    '''
    _energy_TimeDomain_chunk - chunkwise function for energy_TimeDomain

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing DAS data
    time_dim : string
        time dimension of da
    
    Returns
    -------
    da_energy : xr.DataArray
        DataArray containing energy in time domain. Units are V^2 (where V is units of da)
    '''
    
    return (da**2).sum(time_dim, keepdims=True)


def energy_TimeDomain(da, time_dim='time'):
    '''
    energy_TimeDomain - calculate energy in time domain using parsevals theorem
        energy is calculated for each chunk in time_dim
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray containing DAS data
    time_dim : string
        time dimension of da

    Returns
    -------
    da_energy : xr.DataArray
        DataArray containing energy in time domain. Units are V^2 (where V is units of da)
    '''
    # move time_dim to last dimension
    da = da.transpose(..., time_dim)

    # Get number of chunks in each dimension
    original_dims = list(da.dims)

    original_chunksize = dict(zip(original_dims, da.data.chunksize))
    nchunks = []

    for k, single_dim in enumerate(original_dims):
        nchunks.append(da.shape[k]/original_chunksize[single_dim])
    nchunks = dict(zip(original_dims, nchunks))

    # get size of output
    sizes_dict = dict(da.sizes)
    sizes_dict.pop(time_dim)
    output_sizes = list(sizes_dict.values()) + [nchunks[time_dim]]


    # define new chunk sizes
    new_chunk_sizes = {}
    for k, item in enumerate(da.dims):
        if item == time_dim:
            new_chunk_sizes[item] = 1
        else:
            new_chunk_sizes[item] = original_chunksize[item]
    
    # create template for xarray.map_blocks
    template = xr.DataArray(
        dask.array.random.random(
            output_sizes, chunks=list(new_chunk_sizes.values())),
        dims=da.dims,
        name=f'energy in {time_dim} dimension')

    da_energy = da.map_blocks(_energy_TimeDomain_chunk, template=template, kwargs={'time_dim':time_dim})

    return da_energy


# I think everything below this is implemented in xrsignal
def filtfilt(da, dim, **kwargs):
    '''
    filtfilt - this is an implentation of [scipy.signal.fitlfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
    This will filter the DAS data in time for each chunk. This process maps chunks and will therefore have error at the end of chunks in time.

    By default, this does not compute, but generates the task graph

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing DAS data that you want to filter.
    dim : string
        dimension to filter in (should be dimension in da)
    **kwargs : various types
        passed to [scipy.signal.filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
        as per docs, ['x', 'b', and 'a'] are required

    Returns
    -------
    da_filt : xr.DataArray
        filtered data array in time. This does not compute the result, but just the task map in dask
    '''
    kwargs['dim']='time'

    da_filt = da.map_blocks(filtfilt_chunk, kwargs=kwargs, template=da)

    return da_filt


def filtfilt_chunk(da, dim='time', **kwargs):
    '''
    converts dataarray to numpy, sends it to signal.filtfilt and then reinhereits all coordinates

    Parameters
    ----------
    da : xr.DataArray
    dim : string
        dimension to filter over (should be dimension in da)
    **kwargs : various types
        passed to scipy.signal.filtfilt
    '''

    dim_axis = da.dims.index(dim)
    da_np = da.values
    da_filt = signal.filtfilt(x=da_np, axis=dim_axis, **kwargs)

    da_filtx = xr.DataArray(da_filt, dims=da.dims, coords=da.coords, name=da.name, attrs=da.attrs)

    return da_filtx


def spec(da):
    '''
    very quick implementation to calculate spectrogram
        PSD is calculated for every chunk
    
    Currently hardcoded for chunk size of 3000 in time
    Parameters
    ----------
    da : xr.DataArray
        das data to compute spectrogram for
    '''

    template = xr.DataArray(np.ones((int(da.sizes['time']/3000), 513)), dims=['time','frequency']).chunk({'time':1, 'frequency':513})
    return da.map_blocks(__spec_chunk, template=template)


def __spec_chunk(da):
    '''
    compute PSD for single chunk

    Currently hard coded to handle only a time dimension..
    '''
    f, Pxx = signal.welch(da.values, fs=200, nperseg=1024)

    return xr.DataArray(Pxx, dims='frequency', coords={'frequency':f})


def disp_comprate(fk_filter):
    """Display the sizes of the f-k filter matrix (sparse and dense version) and print the compression ratio

    Parameters
    ----------
    fk_filter : sparse.COO
        f-k filter sparse matrix designed by function dsp.hybrid_filter_design()
    """    
    # Print some info about the f-k filter and the sparse matrix compression
    size_sprfilt_coo = fk_filter.data.nbytes/ (1024 ** 3)
    # If the matrix is a scipy.sparse.csr matrix: 
    # size_sprfilt = (fk_filter.data.nbytes + fk_filter.indices.nbytes + fk_filter.indptr.nbytes) / (1024**3)

    densefk_filter = fk_filter.todense() # fk_filter.toarray() 
    sizefilt = densefk_filter.size * densefk_filter.itemsize / (1024**3)

    print(f'The size of the sparse filter is {size_sprfilt_coo:.4f} Gib')
    print(f'The size of the dense filter is {sizefilt:.2f} Gib')
    print(f'The compression ratio is {sizefilt / size_sprfilt_coo:.2f}')
    return