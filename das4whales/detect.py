# Detection module of DAS4whales package
import numpy as np
import scipy.signal as sp

def gen_linear_chirp(fmin, fmax, duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    y = sp.chirp(t, f0=fmax, f1=fmin, t1=duration, method='linear')
    return y


def gen_template_fincall(time, fs, fmin = 15., fmax = 25., duration = 1.):
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

    chirp_signal = gen_linear_chirp(fmin, fmax, duration, fs)
    template = np.zeros(np.shape(time))
    template[:len(chirp_signal)] = chirp_signal * np.hanning(len(chirp_signal))
    return template


def compute_correlation_matrix(data, template):
    # Normalize data along axis 1 by its maximum
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    template = (template - np.mean(template)) / np.std(template)

    # Compute correlation along axis 1
    correlation_matrix = np.zeros_like(data)

    for i in range(data.shape[0]):

        corr = sp.correlate(norm_data[i, :], template, mode='full', method='fft') / len(template)
        correlation_matrix[i, :] = corr[len(corr) // 2 :]

    return correlation_matrix