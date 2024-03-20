import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
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
    bar = fig.colorbar(shw, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope (x$10^{-9}$)')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')
    plt.tight_layout()
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
    roseus = import_roseus()
    fig, ax = plt.subplots(figsize=fig_size)

    shw = ax.pcolormesh(tt, ff, p, shading='auto', cmap=roseus, vmin=v_min, vmax=v_max)
    ax.set_ylim(f_min, f_max)

    # Colorbar
    bar = fig.colorbar(shw, aspect=20)
    bar.set_label('dB (strain x$10^{-9}$)')
    plt.show()


def plot_3calls(channel, time, t1, t2, t3):

    plt.figure(figsize=(12,4))

    plt.subplot(211)
    plt.plot(time, channel, ls='-')
    plt.xlim([time[0], time[-1]])
    plt.ylabel('strain [-]')
    plt.grid()
    plt.tight_layout()

    plt.subplot(234)
    plt.plot(time, channel)
    plt.ylabel('strain [-]')
    plt.xlabel('time [s]')
    plt.xlim([t1, t1+2.])
    plt.grid()
    plt.tight_layout()

    plt.subplot(235)
    plt.plot(time, channel)   
    plt.xlim([t2, t2+2.])
    plt.xlabel('time [s]')
    plt.grid()
    plt.tight_layout()

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
    plt.tight_layout()
    plt.show()

    return


def detection_mf(trace, peaks_idx_HF, peaks_idx_LF, time, dist, fs, dx, selected_channels, file_begin_time_utc=None):
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
    bar = fig.colorbar(cplot, aspect=30, pad=0.015)
    bar.set_label('Strain Envelope [-] (x$10^{-9}$)')
    plt.xlabel('Time [s]')  
    plt.ylabel('Distance [km]')
    plt.legend(loc="upper right")
    # plt.savefig('test.pdf', format='pdf')

    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def snr_matrix(snr_m, time, dist, vmax, file_begin_time_utc=None):
    """Matrix plot of the local signal to noise ratio (SNR)

    Parameters
    ----------
    snr_m : numpy.ndarray
        [channel x time sample] array containing the SNR in the spatio-temporal domain
    time : nummpy.ndarray
        time vector
    dist : numpy.ndarray
        distance vector along the cable
    vmax : float
        maximun value of the plot (dB)
    """    
    fig = plt.figure(figsize=(12, 10))
    snrp = plt.imshow(snr_m, extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='jet', origin='lower',  aspect='auto', vmin=0, vmax=vmax)
    bar = fig.colorbar(snrp, aspect=30, pad=0.015)
    bar.set_label('SNR [dB]')
    bar.ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.0f'))
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [km]')
    
    if isinstance(file_begin_time_utc, datetime):
        plt.title(file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S"), loc='right')

    plt.tight_layout()
    plt.show()

    return


def plot_cross_correlogramHL(corr_m_HF, corr_m_LF, time, dist, maxv, minv=0, file_begin_time_utc=None):
    """
    Plot the cross-correlogram between HF and LF notes.

    Parameters
    ----------
    corr_m_HF : numpy.ndarray
        The cross-correlation matrix of the HF notes.
    corr_m_LF : numpy.ndarray
        The cross-correlation matrix of the LF notes.
    time : numpy.ndarray
        The time values.
    dist : numpy.ndarray
        The distance values.
    maxv : float
        The maximum value for the colorbar.
    minv : int, optional
        The minimum value for the colorbar. Default is 0.
    file_begin_time_utc : datetime.datetime, optional
        The beginning time of the file in UTC. Default is None.

    Returns
    -------
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    im1 = ax1.imshow(abs(sp.hilbert(corr_m_HF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo', origin='lower', aspect='auto', vmin=minv, vmax=maxv) 
    ax1.set_xlabel('Time [s]') 
    ax1.set_ylabel('Distance [km]') 
    ax1.set_title('HF note', loc='right')

    im2 = ax2.imshow(abs(sp.hilbert(corr_m_LF, axis=1)), extent=[time[0], time[-1], dist[0] / 1e3, dist[-1] / 1e3], cmap='turbo',origin='lower', aspect='auto', vmin=minv, vmax=maxv) 
    ax2.set_xlabel('Time [s]') 
    ax2.set_title('LF note', loc='right')

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', aspect=50, pad=0.02) 
    cbar.set_label('Cross-correlation envelope []')
    plt.show()

    return


def import_roseus():
    """
    Import the colormap from the colormap/roseus_matplotlib.py file

    Returns
    -------
    ListedColormap
        colormap
    """
    from matplotlib.colors import ListedColormap
    # Roseus colormap data
    # https://github.com/dofuuz/roseus

    roseus_data = [
        [0.004528, 0.004341, 0.004307],
        [0.005625, 0.006156, 0.006010],
        [0.006628, 0.008293, 0.008161],
        [0.007551, 0.010738, 0.010790],
        [0.008382, 0.013482, 0.013941],
        [0.009111, 0.016520, 0.017662],
        [0.009727, 0.019846, 0.022009],
        [0.010223, 0.023452, 0.027035],
        [0.010593, 0.027331, 0.032799],
        [0.010833, 0.031475, 0.039361],
        [0.010941, 0.035875, 0.046415],
        [0.010918, 0.040520, 0.053597],
        [0.010768, 0.045158, 0.060914],
        [0.010492, 0.049708, 0.068367],
        [0.010098, 0.054171, 0.075954],
        [0.009594, 0.058549, 0.083672],
        [0.008989, 0.062840, 0.091521],
        [0.008297, 0.067046, 0.099499],
        [0.007530, 0.071165, 0.107603],
        [0.006704, 0.075196, 0.115830],
        [0.005838, 0.079140, 0.124178],
        [0.004949, 0.082994, 0.132643],
        [0.004062, 0.086758, 0.141223],
        [0.003198, 0.090430, 0.149913],
        [0.002382, 0.094010, 0.158711],
        [0.001643, 0.097494, 0.167612],
        [0.001009, 0.100883, 0.176612],
        [0.000514, 0.104174, 0.185704],
        [0.000187, 0.107366, 0.194886],
        [0.000066, 0.110457, 0.204151],
        [0.000186, 0.113445, 0.213496],
        [0.000587, 0.116329, 0.222914],
        [0.001309, 0.119106, 0.232397],
        [0.002394, 0.121776, 0.241942],
        [0.003886, 0.124336, 0.251542],
        [0.005831, 0.126784, 0.261189],
        [0.008276, 0.129120, 0.270876],
        [0.011268, 0.131342, 0.280598],
        [0.014859, 0.133447, 0.290345],
        [0.019100, 0.135435, 0.300111],
        [0.024043, 0.137305, 0.309888],
        [0.029742, 0.139054, 0.319669],
        [0.036252, 0.140683, 0.329441],
        [0.043507, 0.142189, 0.339203],
        [0.050922, 0.143571, 0.348942],
        [0.058432, 0.144831, 0.358649],
        [0.066041, 0.145965, 0.368319],
        [0.073744, 0.146974, 0.377938],
        [0.081541, 0.147858, 0.387501],
        [0.089431, 0.148616, 0.396998],
        [0.097411, 0.149248, 0.406419],
        [0.105479, 0.149754, 0.415755],
        [0.113634, 0.150134, 0.424998],
        [0.121873, 0.150389, 0.434139],
        [0.130192, 0.150521, 0.443167],
        [0.138591, 0.150528, 0.452075],
        [0.147065, 0.150413, 0.460852],
        [0.155614, 0.150175, 0.469493],
        [0.164232, 0.149818, 0.477985],
        [0.172917, 0.149343, 0.486322],
        [0.181666, 0.148751, 0.494494],
        [0.190476, 0.148046, 0.502493],
        [0.199344, 0.147229, 0.510313],
        [0.208267, 0.146302, 0.517944],
        [0.217242, 0.145267, 0.525380],
        [0.226264, 0.144131, 0.532613],
        [0.235331, 0.142894, 0.539635],
        [0.244440, 0.141559, 0.546442],
        [0.253587, 0.140131, 0.553026],
        [0.262769, 0.138615, 0.559381],
        [0.271981, 0.137016, 0.565500],
        [0.281222, 0.135335, 0.571381],
        [0.290487, 0.133581, 0.577017],
        [0.299774, 0.131757, 0.582404],
        [0.309080, 0.129867, 0.587538],
        [0.318399, 0.127920, 0.592415],
        [0.327730, 0.125921, 0.597032],
        [0.337069, 0.123877, 0.601385],
        [0.346413, 0.121793, 0.605474],
        [0.355758, 0.119678, 0.609295],
        [0.365102, 0.117540, 0.612846],
        [0.374443, 0.115386, 0.616127],
        [0.383774, 0.113226, 0.619138],
        [0.393096, 0.111066, 0.621876],
        [0.402404, 0.108918, 0.624343],
        [0.411694, 0.106794, 0.626540],
        [0.420967, 0.104698, 0.628466],
        [0.430217, 0.102645, 0.630123],
        [0.439442, 0.100647, 0.631513],
        [0.448637, 0.098717, 0.632638],
        [0.457805, 0.096861, 0.633499],
        [0.466940, 0.095095, 0.634100],
        [0.476040, 0.093433, 0.634443],
        [0.485102, 0.091885, 0.634532],
        [0.494125, 0.090466, 0.634370],
        [0.503104, 0.089190, 0.633962],
        [0.512041, 0.088067, 0.633311],
        [0.520931, 0.087108, 0.632420],
        [0.529773, 0.086329, 0.631297],
        [0.538564, 0.085738, 0.629944],
        [0.547302, 0.085346, 0.628367],
        [0.555986, 0.085162, 0.626572],
        [0.564615, 0.085190, 0.624563],
        [0.573187, 0.085439, 0.622345],
        [0.581698, 0.085913, 0.619926],
        [0.590149, 0.086615, 0.617311],
        [0.598538, 0.087543, 0.614503],
        [0.606862, 0.088700, 0.611511],
        [0.615120, 0.090084, 0.608343],
        [0.623312, 0.091690, 0.605001],
        [0.631438, 0.093511, 0.601489],
        [0.639492, 0.095546, 0.597821],
        [0.647476, 0.097787, 0.593999],
        [0.655389, 0.100226, 0.590028],
        [0.663230, 0.102856, 0.585914],
        [0.670995, 0.105669, 0.581667],
        [0.678686, 0.108658, 0.577291],
        [0.686302, 0.111813, 0.572790],
        [0.693840, 0.115129, 0.568175],
        [0.701300, 0.118597, 0.563449],
        [0.708682, 0.122209, 0.558616],
        [0.715984, 0.125959, 0.553687],
        [0.723206, 0.129840, 0.548666],
        [0.730346, 0.133846, 0.543558],
        [0.737406, 0.137970, 0.538366],
        [0.744382, 0.142209, 0.533101],
        [0.751274, 0.146556, 0.527767],
        [0.758082, 0.151008, 0.522369],
        [0.764805, 0.155559, 0.516912],
        [0.771443, 0.160206, 0.511402],
        [0.777995, 0.164946, 0.505845],
        [0.784459, 0.169774, 0.500246],
        [0.790836, 0.174689, 0.494607],
        [0.797125, 0.179688, 0.488935],
        [0.803325, 0.184767, 0.483238],
        [0.809435, 0.189925, 0.477518],
        [0.815455, 0.195160, 0.471781],
        [0.821384, 0.200471, 0.466028],
        [0.827222, 0.205854, 0.460267],
        [0.832968, 0.211308, 0.454505],
        [0.838621, 0.216834, 0.448738],
        [0.844181, 0.222428, 0.442979],
        [0.849647, 0.228090, 0.437230],
        [0.855019, 0.233819, 0.431491],
        [0.860295, 0.239613, 0.425771],
        [0.865475, 0.245471, 0.420074],
        [0.870558, 0.251393, 0.414403],
        [0.875545, 0.257380, 0.408759],
        [0.880433, 0.263427, 0.403152],
        [0.885223, 0.269535, 0.397585],
        [0.889913, 0.275705, 0.392058],
        [0.894503, 0.281934, 0.386578],
        [0.898993, 0.288222, 0.381152],
        [0.903381, 0.294569, 0.375781],
        [0.907667, 0.300974, 0.370469],
        [0.911849, 0.307435, 0.365223],
        [0.915928, 0.313953, 0.360048],
        [0.919902, 0.320527, 0.354948],
        [0.923771, 0.327155, 0.349928],
        [0.927533, 0.333838, 0.344994],
        [0.931188, 0.340576, 0.340149],
        [0.934736, 0.347366, 0.335403],
        [0.938175, 0.354207, 0.330762],
        [0.941504, 0.361101, 0.326229],
        [0.944723, 0.368045, 0.321814],
        [0.947831, 0.375039, 0.317523],
        [0.950826, 0.382083, 0.313364],
        [0.953709, 0.389175, 0.309345],
        [0.956478, 0.396314, 0.305477],
        [0.959133, 0.403499, 0.301766],
        [0.961671, 0.410731, 0.298221],
        [0.964093, 0.418008, 0.294853],
        [0.966399, 0.425327, 0.291676],
        [0.968586, 0.432690, 0.288696],
        [0.970654, 0.440095, 0.285926],
        [0.972603, 0.447540, 0.283380],
        [0.974431, 0.455025, 0.281067],
        [0.976139, 0.462547, 0.279003],
        [0.977725, 0.470107, 0.277198],
        [0.979188, 0.477703, 0.275666],
        [0.980529, 0.485332, 0.274422],
        [0.981747, 0.492995, 0.273476],
        [0.982840, 0.500690, 0.272842],
        [0.983808, 0.508415, 0.272532],
        [0.984653, 0.516168, 0.272560],
        [0.985373, 0.523948, 0.272937],
        [0.985966, 0.531754, 0.273673],
        [0.986436, 0.539582, 0.274779],
        [0.986780, 0.547434, 0.276264],
        [0.986998, 0.555305, 0.278135],
        [0.987091, 0.563195, 0.280401],
        [0.987061, 0.571100, 0.283066],
        [0.986907, 0.579019, 0.286137],
        [0.986629, 0.586950, 0.289615],
        [0.986229, 0.594891, 0.293503],
        [0.985709, 0.602839, 0.297802],
        [0.985069, 0.610792, 0.302512],
        [0.984310, 0.618748, 0.307632],
        [0.983435, 0.626704, 0.313159],
        [0.982445, 0.634657, 0.319089],
        [0.981341, 0.642606, 0.325420],
        [0.980130, 0.650546, 0.332144],
        [0.978812, 0.658475, 0.339257],
        [0.977392, 0.666391, 0.346753],
        [0.975870, 0.674290, 0.354625],
        [0.974252, 0.682170, 0.362865],
        [0.972545, 0.690026, 0.371466],
        [0.970750, 0.697856, 0.380419],
        [0.968873, 0.705658, 0.389718],
        [0.966921, 0.713426, 0.399353],
        [0.964901, 0.721157, 0.409313],
        [0.962815, 0.728851, 0.419594],
        [0.960677, 0.736500, 0.430181],
        [0.958490, 0.744103, 0.441070],
        [0.956263, 0.751656, 0.452248],
        [0.954009, 0.759153, 0.463702],
        [0.951732, 0.766595, 0.475429],
        [0.949445, 0.773974, 0.487414],
        [0.947158, 0.781289, 0.499647],
        [0.944885, 0.788535, 0.512116],
        [0.942634, 0.795709, 0.524811],
        [0.940423, 0.802807, 0.537717],
        [0.938261, 0.809825, 0.550825],
        [0.936163, 0.816760, 0.564121],
        [0.934146, 0.823608, 0.577591],
        [0.932224, 0.830366, 0.591220],
        [0.930412, 0.837031, 0.604997],
        [0.928727, 0.843599, 0.618904],
        [0.927187, 0.850066, 0.632926],
        [0.925809, 0.856432, 0.647047],
        [0.924610, 0.862691, 0.661249],
        [0.923607, 0.868843, 0.675517],
        [0.922820, 0.874884, 0.689832],
        [0.922265, 0.880812, 0.704174],
        [0.921962, 0.886626, 0.718523],
        [0.921930, 0.892323, 0.732859],
        [0.922183, 0.897903, 0.747163],
        [0.922741, 0.903364, 0.761410],
        [0.923620, 0.908706, 0.775580],
        [0.924837, 0.913928, 0.789648],
        [0.926405, 0.919031, 0.803590],
        [0.928340, 0.924015, 0.817381],
        [0.930655, 0.928881, 0.830995],
        [0.933360, 0.933631, 0.844405],
        [0.936466, 0.938267, 0.857583],
        [0.939982, 0.942791, 0.870499],
        [0.943914, 0.947207, 0.883122],
        [0.948267, 0.951519, 0.895421],
        [0.953044, 0.955732, 0.907359],
        [0.958246, 0.959852, 0.918901],
        [0.963869, 0.963887, 0.930004],
        [0.969909, 0.967845, 0.940623],
        [0.976355, 0.971737, 0.950704],
        [0.983195, 0.975580, 0.960181],
        [0.990402, 0.979395, 0.968966],
        [0.997930, 0.983217, 0.976920],]
    return ListedColormap(roseus_data, name='Roseus')