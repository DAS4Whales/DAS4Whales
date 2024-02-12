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
            
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(fk_filter_matrix, extent=[min(freq),max(freq),min(knum),max(knum)],aspect='auto')
    plt.show()

    return fk_filter_matrix


def generate_hybrid_filter_matrix(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500, fmin=15, fmax=25):
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

    # Find indices corresponding to the wavenumber range of interest
    kmin_idx = np.argmax(knum >= - fmax / cs_min)
    kmax_idx = np.argmax(knum >= fmax / cs_min)

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    # Wave speed is the ratio between the frequency and the wavenumber
    fk_filter_matrix = np.zeros(shape=(len(knum), len(freq)), dtype=float, order='F')

    sos = sp.butter(8,[fmin/(fs/2),fmax/(fs/2)],'bp', output='sos')
    w, h = sp.sosfreqz(sos, worN=len(freq)//2)
    H = np.concatenate([np.flip(np.abs(h)), np.abs(h)])
    
    # fk_filter_matrix = np.tile(H, (len(knum), 1))
    # print(trace_shape)

    # Going through wavenumbers
    for i in range(kmin_idx, kmax_idx):
        # Taking care of very small wavenumber to avoid 0 division
        if abs(knum[i]) < 0.00005:
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
            fk_filter_matrix[i, :] = filter_line * H
            
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.tile(H, (len(knum), 1)), extent=[min(freq),max(freq),min(knum),max(knum)],aspect='auto')
    # plt.plot(freq, np.tile(H, (len(knum), 0))[kmin_idx+10,:])
    plt.show()
    # fk_filter_matrix * 
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
