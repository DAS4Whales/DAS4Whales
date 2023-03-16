import numpy as np
import scipy.signal as sp


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


# Filters
def fk_filtering(trace, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500):
    """
    Designs and apply f-k filtering to DAS strain data
    Keeps by default data with propagation speed [1450-3000] m/s

    The transition band is inspired and adapted from Yi Lin's matlab fk function
    https://github.com/nicklinyi/seismic_utils/blob/master/fkfilter.m

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
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
    :return: trace, a [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal
    domain

    """

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx

    # Get the dimensions of the trace data
    nnx = trace.shape[0]
    nns = trace.shape[1]

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    # Wave speed is the ratio between the frequency and the wavenumber
    fk_filter_matrix = np.ndarray(shape=fk_trace.shape, dtype=float, order='F')

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

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real


def fk_filter(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500):
    """
    Designs a f-k filter for DAS strain data
    Keeps by default data with propagation speed [1450-3400] m/s

    The transition band is inspired and adapted from Yi Lin's matlab fk function
    https://github.com/nicklinyi/seismic_utils/blob/master/fkfilter.m

    Inputs:
    :param trace: a tuple with the dimensions of the strain data in the spatio-temporal domain such as
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

    return fk_filter_matrix


def fk_filt(trace, fk_filter_matrix):
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
