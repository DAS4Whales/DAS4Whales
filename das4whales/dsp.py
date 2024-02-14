import numpy as np
import scipy.signal as sp
import librosa


# Transformations
def get_fx(trace, nfft):
    """
    Apply a fast Fourier transform (fft) to each channel of the strain data matrix

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    :param nfft: number of time samples used for the FFT.

    Outputs:
    :return: trace, a [channel x freq. sample] nparray containing the strain data in the spatio-spectral domain

    """

    fx = 2 * (abs(np.fft.fftshift(np.fft.fft(trace, nfft), axes=1)))
    fx /= nfft
    fx *= 10 ** 9
    return fx


def get_spectrogram(waveform, fs, nfft=128, overlap_pct=0.8):
    """
    Get the spectrogram of a single channel

    Inputs:
    :param waveform: single channel temporal signal
    :param fs: the sampling frequency (Hz)
    :param nfft: number of time samples used for the STFT. Default 128
    :param overlap_pct: percentage of overlap in the spectrogram. Default 0.8

    Outputs:
    :return: a spectrogram and associated time & frequency vectors

    """

    spectrogram = np.abs(librosa.stft(
        y=waveform, n_fft=nfft,
        hop_length=int(np.floor(nfft * (1 - overlap_pct)))))

    # Axis
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]

    tt = np.linspace(0, len(waveform)/fs, num=width)
    ff = np.linspace(0, fs / 2, num=height)

    p = 10 * np.log10(spectrogram * 10 ** 9)

    return p, tt, ff


# Filters
def fk_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500):
    """
    Designs a f-k filter for DAS strain data
    Keeps by default data with propagation speed [1450-3400] m/s

    The transition band is inspired and adapted from Yi Lin's matlab fk function
    https://github.com/nicklinyi/seismic_utils/blob/master/fkfilter.m

    Inputs:
    :param trace_shape: a tuple with the dimensions of the strain data in the spatio-temporal domain such as
    trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    :param selected_channels: a list of the selected channels number  [start, end, step]
    :param dx: the channel spacing (m)
    :param fs: the sampling frequency (Hz)
    :param cs_min: the minimum selected sound speeds for the f-k passband filtering (m/s). Default 1400 m/s
    :param cp_min: the minimum selected sound speed for the f-k stopband filtering, values should frame
    [c_min and c_max] (m/s). Default 1450 m/s.
    :param cp_max: the maximum selected sound speeds for the f-k passband filtering (m/s). Default 3400 m/s
    :param cs_max: the maximumselected sound speed for the f-k stopband filtering, values should frame
    [c_min and c_max] (m/s). Default 3500 m/s

    Outputs:
    :return: fk_filter_matrix, a [channel x time sample] nparray containing the f-k-filter

    """

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx

    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    # Wave speed is the ratio between the frequency and the wavenumber
    fk_filter_matrix = np.ndarray(shape=(len(knum), len(freq)), dtype=float, order='F')

    # Going through wavenumbers
    for i in range(len(knum)):
        # Taking care of very small wavenumber to avoid 0 division
        if abs(knum[i]) < 0.005:
            fk_filter_matrix[i, :] = np.zeros(shape=[len(freq)], dtype=float, order='F')
        else:
            filter_line = np.ones(shape=[len(freq)], dtype=float, order='F')
            speed = abs(freq / knum[i])

            # Filter transition band, ramping up from cs_min to cp_min
            selected_speed_mask = ((speed >= cs_min) & (speed <= cp_min))
            filter_line[selected_speed_mask] = np.sin(0.5 * np.pi *
                                                      (speed[selected_speed_mask] - cs_min) / (cp_min - cs_min))
            # Filter transition band, going down from cp_max to cs_max
            selected_speed_mask = ((speed >= cp_max) & (speed <= cs_max))
            filter_line[selected_speed_mask] = 1 - np.sin(0.5 * np.pi *
                                                          (speed[selected_speed_mask] - cp_max) / (cs_max - cp_max))
            # Stopband
            filter_line[speed >= cs_max] = 0
            filter_line[speed < cs_min] = 0

            # Fill the filter matrix
            fk_filter_matrix[i, :] = filter_line
    
    # Display the filter
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(fk_filter_matrix, extent=[min(freq),max(freq),min(knum),max(knum)],aspect='auto')
    plt.figure()
    plt.plot(freq, fk_filter_matrix[2000, :])
    plt.show()

    return fk_filter_matrix


def hybrid_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400., cp_min=1450., fmin=15., fmax=25., display_filter=False):
    """Designs a bandpass f-k hybrid filter for DAS strain data
        Keeps by default data with propagation speed above 1450 m/s between [15 - 25] Hz (designed for fin whales)

    Parameters
    ----------
    trace_shape : tuple
        tuple with the dimensions of the strain data in the spatio-temporal domain such as trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    selected_channels : list
        list of the selected channels number  [start, end, step]
    dx : float
        channel spacing (m)
    fs : float
        sampling frequency (Hz)
    cs_min : float, optional
        lower minimum selected sound speeds for the f-k highpass filtering (m/s), by default 1400 m/s
    cp_min : float, optional
        higher minimum selected sound speed for the f-k highpass filtering, by default 1450 m/s
    fmin : float, optional
        minimum frequency for the passband, by default 15
    fmax : float, optional
        maximum frequency for the passband, by default 25
    display_filter : bool, optional
        option for filter display, by default False

    Returns
    -------
    fk_filter_matrix : ndarray
        [channel x time sample] nparray containing the f-k-filter
    """    

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # 1st step: frequency bandpass filtering
    H = np.zeros_like(freq)
    # set the width of the frequency range tapers
    df_taper = 5 # Hz
    # Apply it to the frequencies of interest
    fpmax = fmax + df_taper
    fpmin = fmin - df_taper
    # Find the corresponding indexes
    fmin_idx = np.argmax(freq >= - fpmax)
    fmax_idx = np.argmax(freq >= fpmax)

    # Filter transition band, ramping up from -fpmax to -fmax
    lup_mask = ((freq >= -fpmax) & (freq <= -fmax))
    H[lup_mask] = np.sin(0.5 * np.pi *(freq[lup_mask] + fpmax) / (fpmax - fmax))
    # Filter passband
    H[(freq >= -fmax) & (freq <= -fmin)] = 1
    # Filter transition band, ramping down from -fmin to -fpmin
    ldo_mask = ((freq >= -fmin) & (freq <= -fpmin))
    H[ldo_mask] = np.cos(0.5 * np.pi *(freq[ldo_mask] + fmin) / (fpmin - fmin))

    # Filter transition band, ramping up from fpmin to fmin
    rup_mask = ((freq >= fpmin) & (freq <= fmin))
    H[rup_mask] = np.sin(0.5 * np.pi * (freq[rup_mask] - fpmin) / (fmin - fpmin))
    # Filter passband
    H[(freq >= fmin) & (freq <= fmax)] = 1
    # Filter transition band, ramping down from fmax to fpmax
    rdo_mask = ((freq >= fmax) & (freq <= fpmax))
    H[rdo_mask] = np.cos(0.5 * np.pi * (freq[rdo_mask] - fmax) / (fmax - fpmax))

    # Replicate the bandpass frequency response along the k-axis
    fk_filter_matrix = np.tile(H, (len(knum), 1))
    
    # 2nd step: filtering waves whose speeds are below cmin, with a taper between csmin and cpmin
    # Going through frequencies between the considered range of the bandpass filter
    for i in range(fmin_idx, fmax_idx):
        # Initiating filter column to zeros
        filter_col = np.zeros_like(knum)

        # Filter transition bands, ramping up from cs_min to cp_min
        ks = freq[i] / cs_min
        kp = freq[i] / cp_min
        # Remove the NaN created by the implementation when ks == kp
        if ks != kp:
            # f- k+ quadrant 
            selected_k_mask = ((knum >= ks) & (knum <= kp))
            filter_col[selected_k_mask] = np.sin(0.5 * np.pi * (knum[selected_k_mask] - ks) / (kp - ks))

            # f+ k+ quadrant                                             
            selected_k_mask = ((knum >= -ks) & (knum <= -kp))
            filter_col[selected_k_mask] = -np.sin(0.5 * np.pi * (knum[selected_k_mask] + ks) / (kp - ks))

            # f+ k- quadrant
            selected_k_mask = ((-knum >= -ks) & (-knum <= -kp))
            filter_col[selected_k_mask] = np.sin(0.5 * np.pi * (knum[selected_k_mask] - ks) / (kp - ks))

            # f- k - quadrant
            selected_k_mask = ((-knum >= ks) & (-knum <= kp))
            filter_col[selected_k_mask] = -np.sin(0.5 * np.pi * (knum[selected_k_mask] + ks) / (kp - ks))

            # Passbands
            # Negative frequencies (ks is negative):
            filter_col[(knum > kp) & (knum < -kp)] = 1
            # Positive frequencies (ks is positive):
            filter_col[(knum < kp) & (knum > -kp)] = 1

        # Fill the filter matrix by multiplication 
        fk_filter_matrix[:, i] *= filter_col 

    # Filter display, optional
    if display_filter: 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.rc('font', size=20) 
        plt.rc('xtick', labelsize=16)  
        plt.rc('ytick', labelsize=16)

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

        ax1 = plt.subplot(gs[0])
        ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto')
        ax1.set_ylabel('k [m$^{-1}$]')
        ax1.set_xlabel('f [Hz]')
        
        ax2 = plt.subplot(gs[2], sharex=ax1)
        ax2.plot(freq, H, lw=3)
        ax2.set_xlabel('f [Hz]')
        ax2.set_ylabel('Gain []')
        ax2.set_xlim([min(freq), max(freq)])
        ax2.grid()

        ax3 = plt.subplot(gs[1], sharey=ax1)
        ax3.plot(fk_filter_matrix[:, fmin_idx + 250], knum, lw=3)
        ax3.set_xlabel('Gain []')
        ax3.set_ylabel('k [m$^{-1}$]')
        ax3.yaxis.set_label_position("right")
        ax3.set_ylim([min(knum), max(knum)])
        ax3.invert_xaxis()
        ax3.yaxis.tick_right()
        ax3.grid()
        plt.tight_layout()
        plt.show()

    return fk_filter_matrix


def fk_filter_filt(trace, fk_filter_matrix):
    """
    Applies a pre-calculated f-k filter to DAS strain data

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    :param fk_filter_matrix: a [channel x time sample] nparray containing the f-k-filter

    Outputs:
    :return: trace, a [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal
    domain

    """

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real


def butterworth_filter(filterspec, fs):
    """
    Designs and a butterworth filter see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    Apply as.
    trace_filtered = sp.sosfiltfilt(filter_sos, trace_original, axis=1)

    Inputs:
    :param filterspec:
    :param fs:

    Outputs:
    :return: filter_sos: a butterworth filter

    """

    filter_order, filter_critical_freq, filter_type_str = filterspec
    # Build a filter of the desired type
    wn = np.array(filter_critical_freq) / (fs / 2)  # convert to angular frequency

    filter_sos = sp.butter(filter_order, wn, btype=filter_type_str, output='sos')

    return filter_sos
