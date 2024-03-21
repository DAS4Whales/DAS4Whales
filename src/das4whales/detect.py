# Detection module of DAS4whales package
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.stats as st
from tqdm import tqdm

## Matched filter detection functions:

def gen_linear_chirp(fmin, fmax, duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    y = sp.chirp(t, f0=fmax, f1=fmin, t1=duration, method='linear')
    return y


def gen_hyperbolic_chirp(fmin, fmax, duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    y = sp.chirp(t, f0=fmax, f1=fmin, t1=duration, method='hyperbolic')
    return y


def gen_template_fincall(time, fs, fmin = 15., fmax = 25., duration = 1., window=True):
    """ generate template of a fin whale call

    Parameters
    ----------
    time : numpy.ndarray
        time vector
    fs : float
        sampling rate in Hz
    fmin : float, optional
        Minimum frequency, by default 15
    fmax : float, optional
        Maximum frequency, by default 25
    duration : float, optional
        Duration of the chirp signal in seconds, by default 1.
    """
    # 1 Hz frequency buffer to compensate the windowing
    df = 0
    chirp_signal = gen_hyperbolic_chirp(fmin-df, fmax + df, duration, fs)
    template = np.zeros(np.shape(time))
    if window:
        template[:len(chirp_signal)] = chirp_signal * np.hanning(len(chirp_signal))
    else: 
        template[:len(chirp_signal)] = chirp_signal
    return template


def shift_xcorr(x, y):
    """compute the shifted (positive lags only) cross correlation between two 1D arrays

    Parameters
    ----------
    x : numpy.ndarray
        1D array containing signal
    y : numpy.ndarray
        1D array containing signal

    Returns
    -------
    numpy.ndarray
        1D array cross-correlation betweem x and y, only for positive lags
    """
    corr = sp.correlate(x, y, mode='full', method='fft')
    return corr[len(x)-1 :]


def shift_nxcorr(x, y):
    """Compute the shifted (positive lags only) normalized cross-correlation with standard deviation normalization.

    Parameters
    ----------
    x : numpy.ndarray
        first input signal.
    y : numpy.ndarray
        second input signal

    Returns
    -------
    numpy.ndarray
        The normalized cross-correlation between the two signals
    """    

    # Compute cross-correlation
    cross_corr = sp.correlate(x, y, mode='full', method='fft')

    # Normalize using standard deviation
    normalized_corr = cross_corr / (np.std(x) * np.std(y) * len(x))
    
    return normalized_corr[len(x)-1 :]


def compute_cross_correlogram(data, template):
    # Normalize data along axis 1 by its maximum (peak normalization)
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.max(np.abs(data), axis=1, keepdims=True)
    template = (template - np.mean(template)) / np.max(np.abs(template))

    # Compute correlation along axis 1
    cross_correlogram = np.empty_like(data)

    for i in range(data.shape[0]):
        cross_correlogram[i, :] = shift_xcorr(norm_data[i, :], template)

    return cross_correlogram


def pick_times(corr_m, fs, threshold=0.3):
    peaks_indexes_m = []

    for corr in tqdm(corr_m, desc="Processing corr_m"):
        peaks_indexes = sp.find_peaks(abs(sp.hilbert(corr)), prominence=threshold)[0]
        peaks_indexes_m.append(peaks_indexes)
    
    return peaks_indexes_m


def convert_pick_times(peaks_indexes_m):
    peaks_indexes_tp = ([], [])

    for i in range(len(peaks_indexes_m)):
        nb_el = len(peaks_indexes_m[i])
        for j in range(nb_el):
            peaks_indexes_tp[0].append(i)
        for el in peaks_indexes_m[i]:
            peaks_indexes_tp[1].append(el)

    peaks_indexes_tp = np.asarray(peaks_indexes_tp)

## Spectrogram correlation functions:

def finKernelLims(f0, f1, bdwdth):
    """
    Calculate the minimum and maximum kernel limits based on given parameters.

    Parameters:
    f0 (float): The first kernel limit.
    f1 (float): The second kernel limit.
    bdwdth (float): The bandwidth.

    Returns:
    tuple: A tuple containing the minimum and maximum kernel limits.
    """

    ker_1 = 10
    ker_2 = 35
    ker_min = np.min([ker_1, ker_2])
    ker_max = np.max([ker_1, ker_2])
    return ker_min, ker_max


def buildkernel(f0, f1, bdwdth, dur, f, t, samp, plotflag=False, kernel_lims=finKernelLims):
    """
    Calculate kernel and plot.

    Parameters:
    ----------
    f0 : float
        Starting frequency.
    f1 : float
        Ending frequency.
    bdwdth : float
        Frequency width of call.
    dur : float
        Call length (seconds).
    f : np.array
        Vector of frequencies returned from plotwav.
    t : np.array
        Vector of times returned from plotwav.
    samp : float
        Sample rate.
    plotflag : bool, optional
        If True, plots kernel. If False, no plot. Default is False.
    kernel_lims : tuple, optional
        Tuple of minimum kernel range and maximum kernel range. Default is finKernelLims.

    Returns:
    -------
    tvec : numpy.array
        Vector of kernel times.
    fvec_sub : numpy.array
        Vector of kernel frequencies.
    BlueKernel : 2-d numpy.array
        Matrix of kernel values.

    Key variables:
    -------------
    tvec : numpy.array
        Kernel times (seconds).
    fvec : numpy.array
        Kernel frequencies.
    BlueKernel : numpy.array
        Matrix of kernel values.
    """

    # create a time vector of the same length as the call, with the same number of points as the spectrogram
    tvec = np.linspace(0, dur, np.size(np.nonzero((t < dur*8) & (t > dur*7)))) 
    # another way: int(dur * fs / (nperseg * (1-overlap_pct)) + 1)
    # define frequency span of kernel to match spectrogram
    fvec = f 
    # preallocate kernel array
    Kdist = np.zeros((len(fvec), len(tvec))) 
    ker_min, ker_max = kernel_lims(f0, f1, bdwdth)
    
    for j in range(len(tvec)):
        # calculate hat function that is centered on linearly decreasing
        # frequency values for each time in tvec
        x = fvec - (f0 + (tvec[j] / dur) * (f1 - f0))
        Kval = (1 - np.square(x) / (bdwdth * bdwdth)) * np.exp(-np.square(x) / (2 * (bdwdth * bdwdth)))
        # store hat function values in preallocated array
        Kdist[:, j] = Kval 
    
    BlueKernel_full = Kdist
    freq_inds = np.where(np.logical_and(fvec >= ker_min, fvec <= ker_max))
    
    fvec_sub = fvec[freq_inds]
    BlueKernel = BlueKernel_full[freq_inds, :][0]
    
    if plotflag:
        plt.figure(figsize=(20, 16))
        plt.pcolormesh(tvec, fvec_sub, BlueKernel, cmap="bwr", vmin=-np.max(np.abs(BlueKernel)), vmax=np.max(np.abs(BlueKernel)))      
        plt.axis([0, dur, np.min(fvec), np.max(fvec)])
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.ylim(ker_min, ker_max)
        plt.title('Fin whale call kernel')
        plt.show()
        
    return tvec, fvec_sub, BlueKernel, freq_inds
    
    return peaks_indexes_tp