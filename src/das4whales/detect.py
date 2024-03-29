# Detection module of DAS4whales package
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.stats as st
from tqdm import tqdm
from das4whales.plot import import_roseus 

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

    return peaks_indexes_tp

## Spectrogram correlation functions:

def get_sliced_nspectrogram(trace, fs, fmin, fmax, nperseg, nhop, plotflag=False):
    """
    Compute the sliced non-stationary spectrogram of a given trace.

    Parameters
    ----------
    trace : numpy.ndarray
        The input trace signal.
    fs : float
        The sampling rate of the trace signal.
    fmin : float
        The minimum frequency of interest.
    fmax : float
        The maximum frequency of interest.
    nperseg : int
        The length of each segment for the spectrogram computation.
    nhop : int
        The number of samples to advance between segments.
    plotflag : bool, optional
        Whether to plot the spectrogram, defaults to False.

    Returns
    -------
    spectrogram : numpy.ndarray
        The computed spectrogram.
    ff : ndarray
        The frequency values of the spectrogram.
    tt : ndarray
        The time values of the spectrogram.

    Notes
    -----
    This function computes the non-stationary spectrogram of a given trace signal.
    The spectrogram is computed using the Short-Time Fourier Transform (STFT) with
    a specified segment length and hop size. The resulting spectrogram is then sliced
    between the specified minimum and maximum frequencies of interest.

    Examples
    --------
    >>> trace = np.random.randn(1000)
    >>> fs = 1000
    >>> fmin = 10
    >>> fmax = 100
    >>> nperseg = 256
    >>> nhop = 128
    >>> spectrogram, ff, tt = get_sliced_nspectrogram(trace, fs, fmin, fmax, nperseg, nhop, plotflag=True)
    """

    spectrogram = np.abs(librosa.stft(y=trace, n_fft=nperseg, hop_length=nhop))
    # Axis
    nf, nt = spectrogram.shape
    tt = np.linspace(0, len(trace)/fs, num=nt)
    ff = np.linspace(0, fs / 2, num=nf)
    p = spectrogram # / np.max(spectrogram)

    # Slice the spectrogram betweem fmin and fmax
    ff_idx = np.where((ff >= fmin) & (ff <= fmax))
    p = p[ff_idx]
    ff = ff[ff_idx]

    if plotflag:
        roseus = import_roseus()
        fig, ax = plt.subplots(figsize=(12,4))
        shw = ax.pcolormesh(tt, ff, p, cmap=roseus, vmin=None, vmax=None)
        # Colorbar
        bar = fig.colorbar(shw, aspect=20, pad=0.015)
        bar.set_label('Normalized amplitude [-]')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()
    
    return p, ff, tt


def buildkernel(f0, f1, bdwdth, dur, f, t, samp, fmin, fmax, plotflag=False):
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
    ker_min, ker_max = fmin, fmax
    
    for j in range(len(tvec)):
        # calculate hat function that is centered on linearly decreasing
        # frequency values for each time in tvec
        # Linearly decreasing frequency values
        x = fvec - (f0 + (tvec[j] / dur) * (f1 - f0))
        # Hyperbolic decreasing frequency values
        # x = fvec - (f0 * f1 * dur / ((f0 - f1) * (tvec[j] / dur) + f1 * dur))
        Kval = (1 - np.square(x) / (bdwdth * bdwdth)) * np.exp(-np.square(x) / (2 * (bdwdth * bdwdth)))
        # store hat function values in preallocated array
        Kdist[:, j] = Kval 
    BlueKernel = Kdist
    # freq_inds = np.where(np.logical_and(fvec >= ker_min, fvec <= ker_max))
    
    # fvec_sub = fvec[freq_inds]
    # BlueKernel = BlueKernel_full[freq_inds, :][0]
    
    if plotflag:
        plt.figure(figsize=(20, 16))
        plt.pcolormesh(tvec, fvec, BlueKernel, cmap="RdBu_r", vmin=-np.max(np.abs(BlueKernel)), vmax=np.max(np.abs(BlueKernel)),)
        plt.axis([0, dur, np.min(fvec), np.max(fvec)])
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.ylim(ker_min, ker_max)
        plt.gca().set_aspect('equal')
        plt.title('Fin whale call kernel')
        plt.show()
        
    return tvec, fvec, BlueKernel


def nxcorr2d(spectro, kernel):
    """
    Calculate the normalized cross-correlation between a spectrogram and a kernel.

    Parameters
    ----------
    spectro : numpy.ndarray
        The spectrogram array.
    kernel : numpy.ndarray
        The kernel array.

    Returns
    -------
    numpy.ndarray
        The maximum correlation values along the time axis.

    Notes
    -----
    The normalized cross-correlation is calculated using `scipy.signal.correlate2d`.
    The correlation values are normalized by dividing by the standard deviation of the spectrogram and the kernel,
    multiplied by the number of columns in the spectrogram.

    Examples
    --------
    >>> spectro = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> kernel = np.array([[1, 0], [0, 1]])
    >>> nxcorr2d(spectro, kernel)
    array([0.        , 0.33333333, 0.66666667])
    """
    correlation = sp.correlate2d(spectro, kernel, mode='same') / (np.std(spectro) * np.std(kernel) * spectro.shape[1])
    maxcorr_t = np.max(correlation, axis=0)

    return maxcorr_t


def xcorr2d(spectro, kernel):
    """
    Calculate the 2D cross-correlation between a spectrogram and a kernel.

    Parameters
    ----------
    spectro : numpy.ndarray
        The input spectrogram array.

    kernel : numpy.ndarray
        The kernel array used for cross-correlation.

    Returns
    -------
    numpy.ndarray
        The resulting cross-correlation array.

    """
    correlation = sp.correlate2d(spectro, kernel, mode='same')
    maxcorr_t = np.max(correlation, axis=0)

    return maxcorr_t


def compute_cross_correlogram_spectrocorr(data, fs, flims, win_size, overlap_pct):
    nperseg = int(win_size * fs)
    nhop = int(np.floor(nperseg * (1 - overlap_pct)))
    noverlap = nperseg - nhop
    print(f'nperseg: {nperseg}, noverlap: {noverlap}, hop_length: {nhop}')   
    fmin, fmax = flims

    # Call metrics from the OOI dataset calls 2021-11-04T020002 
    f0 = 28.
    f1 = 19. # 17.8
    duration = 0.68
    bandwidth = 3 # or 5?

    # Compute correlation along axis 1
    cross_correlogram = None
    kernel = None

    for i in tqdm(range(data.shape[0])):
        spectro, ff, tt = get_sliced_nspectrogram(data[i, :], fs, fmin, fmax, nperseg, nhop, plotflag=False)
        if cross_correlogram is None:
            cross_correlogram = np.empty((data.shape[0], len(tt)))
        if kernel is None:
            tvec, fvec_sub, kernel = buildkernel(f0, f1, bandwidth, duration, ff, tt, fs, fmin, fmax, plotflag=False)
        cross_correlogram[i, :] = xcorr2d(spectro, kernel)

    return cross_correlogram