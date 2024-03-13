# Detection module of DAS4whales package
import numpy as np
import scipy.signal as sp
import scipy.stats as st
from tqdm import tqdm

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
    
    return peaks_indexes_tp