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
        template[:len(chirp_signal)] = chirp_signal * np.hamming(len(chirp_signal))
    else: 
        template[:len(chirp_signal)] = chirp_signal
    return template


def compute_correlation_matrix(data, template):
    # Normalize data along axis 1 by its maximum
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.max(np.abs(data), axis=1, keepdims=True)
    template = (template - np.mean(template)) / np.max(np.abs(template))

    # Compute correlation along axis 1
    correlation_matrix = np.zeros_like(data)

    for i in range(data.shape[0]):
        corr = sp.correlate(norm_data[i, :], template, mode='full', method='fft')
        correlation_matrix[i, :] = corr[len(corr) // 2 :]

    return correlation_matrix


def pick_times(corr_m, IPI, fs, threshold=0.3):
    peaks_indexes_m = []

    for corr in tqdm(corr_m, desc="Processing corr_m"):
        # slow : sp.argrelmax(corr, order=int(IPI * fs))
        # dB values : sp.find_peaks(20 * np.log10(abs(sp.hilbert(corr))), prominence=threshold)[0]
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