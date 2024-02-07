import matplotlib.pyplot as plt
import numpy as np
from das4whales.dsp import get_fx
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