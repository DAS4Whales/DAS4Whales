import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from das4whales.dsp import get_fx, instant_freq
from datetime import datetime


def plot_tx(trace, time, dist, file_begin_time_utc=0, fig_size=(12, 10), v_min=None, v_max=None):
    """
    Spatio-temporal representation (t-x plot) of the strain data

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    :param time: the corresponding time vector
    :param dist: the corresponding distance along the FO cable vector
    :param file_begin_time_utc: the time stamp of the represented file
    :param fig_size: Tuple of the figure dimensions. Default fig_size=(12, 10)
    :param v_min: sets the min nano strain amplitudes of the colorbar. Default v_min=0
    :param v_max: sets the max nano strain amplitudes of the colorbar, Default v_max=0.2

    Outputs:
    :return: a tx plot

    """

    fig = plt.figure(figsize=fig_size)
    #TODO determine if the envelope should be implemented here rather than just abs
    # Replace abs(trace) per abs(sp.hilbert(trace, axis=1)) ? 
    shw = plt.imshow(abs(trace) * 10 ** 9, extent=[time[0], time[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                     origin='lower', cmap='jet', vmin=v_min, vmax=v_max)
    plt.ylabel('Distance (km)')
    plt.xlabel('Time (s)')
    bar = fig.colorbar(shw, aspect=20)
    bar.set_label('Strain (x$10^{-9}$)')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.show()


def plot_fx(trace, dist, fs, file_begin_time_utc=0, win_s=2, nfft=4096, fig_size=(12, 10), f_min=0,
            f_max=100, v_min=None, v_max=None):
    """
    Spatio-spectral (f-k plot) of the strain data

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    :param dist: the corresponding distance along the FO cable vector
    :param fs: the sampling frequency (Hz)
    :param file_begin_time_utc: the time stamp of the represented file
    :param win_s: the duration of each f-k plot (s). Default 2 s
    :param nfft: number of time samples used for the FFT. Default 4096
    :param fig_size: Tuple of the figure dimensions. Default fig_size=(12, 10)
    :param f_min: displayed minimum frequency interval (Hz). Default 0 Hz
    :param f_max: displayed maxumum frequency interval (Hz). Default 100 Hz
    :param v_min: set the min nano strain amplitudes of the colorbar.
    :param v_max: set the max nano strain amplitudes of the colorbar.

    Outputs:
    :return: fx plot

    """

    # Evaluate the number of subplots
    nb_subplots = int(np.ceil(trace.shape[1] / (win_s * fs)))

    # Create the frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # Prepare the plot
    rows = 3
    cols = int(np.ceil(nb_subplots/rows))

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    # Run through the data
    for ind in range(nb_subplots):
        fx = get_fx(trace[:, int(ind * win_s * fs):int((ind + 1) * win_s * fs):1], nfft)
        # fx = np.transpose(fx) - np.mean(fx, axis=1)
        # fx = np.transpose(fx)

        # Plot
        r = ind // cols
        c = ind % cols
        ax = axes[r][c]

        shw = ax.imshow(fx, extent=[freq[0], freq[-1], dist[0] * 1e-3, dist[-1] * 1e-3], aspect='auto',
                        origin='lower', cmap='jet', vmin=v_min, vmax=v_max)

        ax.set_xlim([f_min, f_max])
        if r == rows-1:
            ax.set_xlabel('Frequency (Hz)')
        else:
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)

        if c == 0:
            ax.set_ylabel('Distance (km)')
        else:
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    # Colorbar
    bar = fig.colorbar(shw, ax=axes.ravel().tolist())
    bar.set_label('Strain (x$10^{-9}$)')
    plt.show()


def plot_spectrogram(p, tt, ff, fig_size=(25, 5), v_min=None, v_max=None, f_min=None, f_max=None):
    """

    :param p: spectrogram values in dB
    :param tt: associated time vector (s)
    :param ff: associated frequency vector (Hz)
    :param fig_size: Tuple of the figure dimensions. Default fig_size=(12, 10)
    :param v_min: set the min dB strain amplitudes of the colorbar.
    :param v_max: set the max dB strain amplitudes of the colorbar.
    :param f_min: minimum frequency for the spectrogram display
    :param f_max: maximum frequency for the spectrogram display

    :return:

    """
    fig, ax = plt.subplots(figsize=fig_size)

    shw = ax.pcolormesh(tt, ff, p, cmap="jet", vmin=v_min, vmax=v_max)
    ax.set_ylim(f_min, f_max)

    # Colorbar
    bar = fig.colorbar(shw, aspect=20)
    bar.set_label('Strain (x$10^{-9}$)')
    plt.show()


def plot_3calls(channel, time, t1, t2, t3):

    plt.figure(figsize=(12,4))

    plt.subplot(211)
    plt.plot(time, channel, ls='-')
    plt.xlim([time[0], time[-1]])
    plt.ylabel('strain [-]')
    plt.grid()

    plt.subplot(234)
    plt.plot(time, channel)
    plt.ylabel('strain [-]')
    plt.xlabel('time [s]')
    plt.xlim([t1, t1+2.])
    plt.grid()

    plt.subplot(235)
    plt.plot(time, channel)   
    plt.xlim([t2, t2+2.])
    plt.xlabel('time [s]')
    plt.grid()

    plt.subplot(236)
    plt.plot(time, channel)   
    plt.xlim([t3, t3+2.])
    plt.xlabel('time [s]')
    plt.grid()
    plt.tight_layout()

    # plt.savefig('3calls.pdf', format='pdf')
    plt.show()

    return


def design_mf(trace, hnote, lnote, th, tl, time, fs):
    """Plot to design the matched filter 

    Parameters
    ----------
    trace : numpy.ndarray
        1D time series channel trace
    hnote : numpy.ndarray
        1D time series high frequency note template
    lnote : numpy.ndarray
        1D time series low frequency note template
    th : float
        start time of the high frequency note
    tl : float
        start time of the low frequency note
    time : numpy.ndarray
        1D vector of time values
    fs : float
        sampling frequency
    """    

    nf = int(th * fs)
    nl = int(tl * fs)
    # Create a dummy channel made of two notes at given times (not robust)
    dummy_chan = np.zeros_like(hnote)
    dummy_chan[nf:] = hnote[:-nf]
    dummy_chan[nl:] = lnote[:-nl]

    # Matched filter instantaneous freq
    fi = instant_freq(trace, fs)
    fi_mf = instant_freq(dummy_chan, fs)

    # Plot the generated linear chirp signal
    plt.figure(figsize=(18, 4))
    plt.subplot(121)
    plt.plot(time, (trace) / (np.max(abs(trace))), label='normalized measured fin call')
    plt.plot(time, (dummy_chan) / (np.max(abs(dummy_chan))), label='template')
    plt.title('fin whale call template - HF note')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim(th-0.5, th+1.5)
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(time[1:], fi, label='measured fin call')
    plt.plot(time[1:], fi_mf, label='template')
    plt.xlim([th-0.5, th+1.5])
    plt.ylim([15., 35])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Instantaneous frequency [Hz]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.subplot(121)
    plt.plot(time, (trace - np.mean(trace)) / (np.max(abs(trace))), label='normalized measured fin call')
    plt.plot(time, (dummy_chan - np.mean(dummy_chan)) / (np.max(abs(dummy_chan))), label='template')
    plt.title('fin whale call template - LF note')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim([tl-0.5, tl+1.5])
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(time[1:], fi, label='measured fin call')
    plt.plot(time[1:], fi_mf, label='template')
    plt.xlim([tl-0.5, tl+1.5])
    plt.ylim([12., 28.])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Instantaneous frequency [Hz]')
    plt.legend()
    plt.grid()
    plt.show()

    return


def detection_mf(trace, peaks_idx_HF, peaks_idx_LF, time, dist, fs, dx, selected_channels, file_begin_time_utc=0):
    """Plot the strain trace matrix [dist x time] with call detection above it

    Parameters
    ----------
    trace : numpy.ndarray
        [channel x time sample] array containing the strain data in the spatio-temporal domain
    peaks_idx_HF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the high frequency call
    peaks_idx_LF : tuple
        tuple of lists containing the detected call indexes coordinates (first list: channel idx, second list: time idx) for the low frequency call
    time : numpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    fs : float
        sampling frequency
    dx : float
        spatial step
    selected_channels : list
        list of selected channels indexes [start, stop, step]
    file_begin_time_utc : int, optional
        time stamp of file, by default 0
    """    

    fig = plt.figure(figsize=(12,10))
    cplot = plt.imshow(abs(sp.hilbert(trace, axis=1)) * 1e9, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=0.4, alpha=0.35)
    plt.scatter(peaks_idx_HF[1] / fs, (peaks_idx_HF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='red', marker='x', label='HF_note')
    plt.scatter(peaks_idx_LF[1] / fs, (peaks_idx_LF[0] * selected_channels[2] + selected_channels[0]) * dx /1e3, color='green', marker='.', label='LF_note')
    bar = fig.colorbar(cplot, aspect=20)
    bar.set_label('Strain [-] (x$10^{-9}$)')
    plt.xlabel('Time [s]')  
    plt.ylabel('Distance [km]')
    plt.legend()
    # plt.savefig('test.pdf', format='pdf')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.show()

    return