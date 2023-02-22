import matplotlib.pyplot as plt
import numpy as np
from das4whales.transform import get_fx


def plot_tx(trace, tx, dist, fs, selected_channels, file_begin_time_utc):
    """
    TX plot of the strain data

    Inputs:
    - trace, a channel x sample nparray containing the strain data
    - cmin and cmax: the selected soundspeeds for the f-k "bandpass" filtering

    Outputs:
    - trace, a channel x sample nparray containing the f-k-filtered strain data

    """

    fig = plt.figure(figsize=(12, 10))
    shw = plt.imshow(abs(trace) * 10 ** 9, extent=[tx[0], tx[-1], dist[0] * 1e-3, dist[-1] * 1e-3, ], aspect='auto',
                     origin='lower', cmap='jet', vmin=0, vmax=0.2)
    plt.ylabel('Distance (km)')
    plt.xlabel('Time (s)')
    bar = plt.colorbar(shw, aspect=20)
    bar.set_label('Strain (x$10^{-9}$)')

    plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')


def plot_fx(trace, dist, fs, fileBeginTimeUTC, plotsavefolder, win_s=2, nfft=4096):
    ### Save 2s-long FX-plot of the FK filtered data
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # Run through the data
    for ind in range(int(np.floor(trace.shape[1] / (win_s * fs))) - 1):
        fx = get_fx(trace[:, int(ind * win_s * fs):int((ind + 1) * win_s * fs):1], nfft)

        # Plot
        fig = plt.figure(figsize=(12, 10))
        shw = plt.imshow(fx, extent=[freq[0], freq[-1], dist[0] * 1e-3, dist[-1] * 1e-3], aspect='auto',
                         origin='lower', cmap='jet', vmin=0, vmax=0.03)

        plt.xlim([0, 150])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Distance (km)')

        # Colorbar
        bar = plt.colorbar(shw, aspect=50)
        bar.set_label('Strain (x$10^{-9}$)')

        # Title
        save_time = fileBeginTimeUTC + timedelta(seconds=(ind * win_s))
        plt.title(save_time.strftime("%Y-%m-%d %H:%M:%S"), loc='right')
        plt.tight_layout()

