import matplotlib.pyplot as plt
import numpy as np
from das4whales.dsp import get_fx


def plot_tx(trace, time, dist, file_begin_time_utc, v_min=0, v_max=0.2):
    """
    Spatio-temporal representation (t-x plot) of the strain data

    Inputs:
    - trace, a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    - tx, the corresponding time vector
    - dist, the corresponding distance along the FO cable vector
    - file_begin_time_utc, the time stamp of the represented file
    - v_min and v_max, set the min and max nano strain amplitudes of the colorbar


    """

    fig = plt.figure(figsize=(12, 10))
    shw = plt.imshow(abs(trace) * 10 ** 9, extent=[time[0], time[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                     origin='lower', cmap='jet', vmin=v_min, vmax=v_max)
    plt.ylabel('Distance (km)')
    plt.xlabel('Time (s)')
    bar = fig.colorbar(shw, aspect=20)
    bar.set_label('Strain (x$10^{-9}$)')

    plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')
    plt.show()


def plot_fx(trace, dist, fs, win_s=2, nfft=4096, f_min=0, f_max=100, v_min=0, v_max=0.1):
    """
    Spatio-spectral (f-k plot) of the strain data

    Inputs:
    - trace, a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    - dist, the corresponding distance along the FO cable vector
    - fs, the sampling frequency (Hz)
    - win_s, the duration of each f-k plot (s)
    - nfft, number of time samples used for the FFT
    - f_min=0, f_max=200, displayed frequency interval (Hz)
    - v_min=0, v_max=0.03, set the min and max nano strain amplitudes of the colorbar
    - file_begin_time_utc, the time stamp of the represented file


    """
    # Evaluate the number of subplots
    nb_subplots = int(np.ceil(trace.shape[1] / (win_s * fs)))

    # Create the frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # Prepare the plot
    rows = 3
    cols = int(np.ceil(nb_subplots/rows))

    fig, axes = plt.subplots(rows, cols, figsize=(8, 10))
    # Run through the data
    for ind in range(nb_subplots):
        fx = get_fx(trace[:, int(ind * win_s * fs):int((ind + 1) * win_s * fs):1], nfft)

        # Plot
        r = ind // cols
        c = ind % cols
        ax = axes[r][c]

        shw = ax.imshow(fx, extent=[freq[0], freq[-1], dist[0] * 1e-3, dist[-1] * 1e-3], aspect='auto',
                        origin='lower', cmap='jet', vmin=v_min, vmax=v_max)

        ax.set_xlim([f_min, f_max])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Distance (km)')

    # Colorbar
    bar = fig.colorbar(shw, ax=axes.ravel().tolist())
    bar.set_label('Strain (x$10^{-9}$)')
    plt.show()

