import numpy as np
from librosa import zero_crossings as zc
from scipy import signal
import pywt

from waves.misc import flatten


def calc_univar_bhatt_coeff(x1, x2, n_bins=20):
    """
    
    Compute the Bhattacharyya coefficient of two datasets for single variable.

    The Bhattacharyya coefficient is a measure of the similarity between 
    two probability density functions (PDFs). It quantifies the overlapping 
    area of the histograms constructed x1 and x2.

    Parameters
    ----------
    x1 : 1D array-like
        Data for distribution 1
    x2 : 1D array-like
        Data for distriution 2
    n_bins : int, optional
        Number of bins for histogram. Default is 20.

    Returns
    -------
    bht : float
        The Bhattacharyya coefficient value.

    Example
    -------
        x1 = [1.2, 3.5, 2.1, 4.9, 2.5]
        x2 = [2.0, 4.1, 2.8, 3.2, 1.7]
        bhatt_coeff = compute_bhatt_coeff(x1, x2)
        print(bhatt_coeff)  
        # Output will be a floating-point number between 0 and 1.
   
    """

    # Combine the datasets to calculate shared histogram bounds
    cx = np.concatenate((x1, x2))
    data_min, data_max = min(cx), max(cx)

    # Create normalized histograms (probability density functions)
    h1, bin_edges = np.histogram(x1, bins=n_bins, range=(data_min, data_max),
                                 density=True)
    h2, _ = np.histogram(x2, bins=bin_edges, density=True)

    # Calculate Bhattacharyya coefficient from bin densities
    bht = 0
    for i in range(n_bins):
        p1 = h1[i] * (bin_edges[i+1] - bin_edges[i])
        p2 = h2[i] * (bin_edges[i+1] - bin_edges[i])
        bht += np.sqrt(p1 * p2)

    return bht


def calc_wavelet_packet_decomposition(waveform):
    """
    
    Calculates wavelet packet decomposition.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
        
    Returns
    -------
    partial_energies : float
        Distribution of energy.
        
    """
    E_tot = np.sum(np.power(waveform,2))
    wp = pywt.WaveletPacket(data=waveform, wavelet = 'db2', mode = 'zero',
                            maxlevel=3)
    leaf_names = [n.path for n in wp.get_leaf_nodes(True)]
    decomposed_signals = [wp[path].data for path in leaf_names]
    partial_energies = [np.sum(np.power(decomp,2))/E_tot for decomp 
                        in decomposed_signals]

    return partial_energies


def calc_peak_freq(waveform, dt=10**-7, low_pass=None, high_pass=None):
    """
    
    Calculates peak frequency of waveform.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    low_pass : int, optional
        Low pass filter threshold [Hz]. The default is None.
    high_pass : int, optional
        High pass filter threshold [Hz]. The default is None.
        
    Returns
    -------
    peak_freq : float
        Frequency of maximum FFT amplitude [Hz].
        
    """
    w, z = calc_fft(waveform, dt, low_pass, high_pass)
    max_index = np.argmax(z)
    peak_freq = w[max_index]

    return peak_freq


def calc_freq_centroid(waveform, dt=10**-7, low_pass=None, high_pass=None):
    """
    
    Get frequency centroid of signal. By doing fft first then computing.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    low_pass : int, optional
        Low pass filter threshold [Hz]. The default is None.
    high_pass : int, optional
        High pass filter threshold [Hz]. The default is None.

    Returns
    -------
    freq_centroid : float
        Frequency centroid of signal.

    """
    w, z = calc_fft(waveform, dt, low_pass, high_pass)
    freq_centroid = np.sum(z*w)/np.sum(z)
    
    return freq_centroid


def calc_fft(waveform, dt=10**-7, low_pass=None, high_pass=None):
    """
    
    Performs FFT on waveform.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    low_pass : int, optional
        Low pass filter threshold [Hz]. The default is None.
    high_pass : int, optional
        High pass filter threshold [Hz]. The default is None.
        
    Returns
    -------
    w : array-like
        Frequency.
    z : array-like
        Amplitude.

    """
    z = np.abs(np.fft.fft(waveform))
    w = np.fft.fftfreq(len(z), dt)
    w = w[np.where(w>=0)] 
    z = z[np.where(w>=0)] 
    if low_pass is not None:
        z = z[np.where(w > low_pass)]
        w = w[np.where(w > low_pass)]
    if high_pass is not None:
        z = z[np.where(w < high_pass)]
        w = w[np.where(w < high_pass)]

    return w, z


def calc_max_amplitude(waveform):
    """
    
    Calculate max amplitude value for waveform.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.

    Returns
    -------
    max_amp : float
        Maximum waveform amplitude [V].

    """   
    max_amp = np.max(waveform)
    
    return max_amp


def calc_rise_freq(waveform, dt=10**-7, threshold=0.1):
    """
    
    Calculate rise frequency.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Returns
    -------
    rise_freq : float
        Average frequency of the rising part of the wave.

    """        
    imin, _ = calc_signal_start_end(waveform, threshold=threshold)
    risingpart = waveform[imin:np.argmax(waveform)]
    zero_crossings = calc_zero_crossings(risingpart, threshold=threshold)
    rise_freq = zero_crossings/(len(risingpart)*dt)
    
    return rise_freq
    

def calc_weighted_peak_freq(waveform, dt=10**-7, low_pass=None, 
                            high_pass=None):
    """
    
    Calculates weighted peak frequency of waveform.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    low_pass : int, optional
        Low pass filter threshold [Hz]. The default is None.
    high_pass : int, optional
        High pass filter threshold [Hz]. The default is None.
        
    Returns
    -------
    wpf : float
        Weighted peak frequency.
        
    """
    freq_centroid = calc_freq_centroid(waveform, dt=dt, low_pass=low_pass, 
                                       high_pass=high_pass)
    peak_freq = calc_peak_freq(waveform, dt=dt, low_pass=low_pass, 
                                       high_pass=high_pass)
    wpf = np.sqrt(freq_centroid*peak_freq)
    
    return wpf


def calc_reverb_freq(waveform, dt=10**-7, threshold=0.1):
    """
    
    Calculate reverb frequency.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Returns
    -------
    reverb_freq : float
        Average frequency of the falling part of the wave.

    """        
    _, imax = calc_signal_start_end(waveform, threshold=threshold)
    fallingpart = waveform[np.argmax(waveform):imax] 
    zero_crossings = calc_zero_crossings(fallingpart, threshold=threshold)
    reverb_freq = zero_crossings/(len(fallingpart)*dt)
    
    return reverb_freq
    

def calc_decay_time(waveform, threshold=0.1):
    """
    
    Calculate decay time for waveform.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Returns
    -------
    decay_time : float
        Time from peak till waveform end [us].
        
    """
    _, imax = calc_signal_start_end(waveform, threshold=threshold)
    end_time = imax/10 
    peak_time = calc_peak_time(waveform)
    decay_time = end_time-peak_time
    
    return decay_time


def calc_peak_time(waveform):
    """
    
    Calculate time of max amplitude value for waveform.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.

    Returns
    -------
    peak_time : float
        Maximum waveform amplitude time [us].

    """   
    peak_time = np.argmax(waveform) / 10
    
    return peak_time


def calc_norm_fft(waveform, dt, low_pass=None, high_pass=None):
    """
    
    Performs FFT on waveform. Normalizes.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    low_pass : int, optional
        Low pass filter threshold [Hz]. The default is None.
    high_pass : int, optional
        High pass filter threshold [Hz]. The default is None.
        
    Returns
    -------
    w : array-like
        Frequency.
    z : array-like
        Normalized amplitude. 

    """
    w,z = calc_fft(waveform, dt, low_pass, high_pass)
    z=z/max(z)
    
    return w, z


def calc_partial_power(waveform, dt = 10**-7, lower_bound=None, 
                     upper_bound=None):
    """
    
    Gets partial power of signal from waveform from lower to upper freq.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    lower_bound : float, optional
        Lower bound of partial power in Hz. The default is None.
    upper_bound : float, optional
        Upper bound of partial power in Hz. The default is None.

    Returns
    -------
    partial_power : float
        Area under fft for specified frequency bounds.

    """
    w, z = calc_fft(waveform, dt)
    total_power = np.sum(z)
    power = np.sum(z[np.where((w>=lower_bound) & (w<upper_bound))])
    partial_power = power/total_power

    return partial_power


def calc_zero_crossings(waveform, threshold=0.1):
    """
    
    Gets number of zero crossings in an AE signal.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Returns
    -------
    num_zero_crossings : int
        Average frequency of signal in Hz.

    """
    imin, imax = calc_signal_start_end(waveform, threshold=threshold)
    cut_signal = waveform[imin:imax]
    zero_crossings = len(np.nonzero(zc(cut_signal))[0])
    
    return zero_crossings


def calc_signal_start_end(waveform, threshold=0.1):
    """
    
    Gets indices of the signal start and end according to max amplitude perc.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Raises
    ------
    ValueError
        Threshold must be between 0 and 1.

    Returns
    -------
    start_index : int
        Array index of signal start.
    end_index : int
        Array index of signal signal end.

    """
    if threshold<0 or threshold>1:
        raise ValueError('Threshold must be between 0 and 1')
    max_amp = np.max(waveform)
    start_index, end_index = \
        np.nonzero(waveform > threshold*max_amp)[0][[0, -1]]
        
    return start_index, end_index


def calc_signal_energy(waveform):
    """
    
    Calculate signal energy. Sum of squares.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.

    Returns
    -------
    energy : float
        Area under squared magnitude of signal.

    """
    energy = 0
    for x in waveform:
        energy += x**2
        
    return energy


def calc_duration(waveform, threshold=0.1):
    """
    
    Calculates duration of waveform in microseconds.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.
        
    Returns
    -------
    duration : float
        Signal duration [us].

    """
    imin, imax = calc_signal_start_end(waveform, threshold=threshold)
    start_time = imin/10 
    end_time = imax/10
    duration = end_time-start_time
    
    return duration


def calc_rise_time(waveform, threshold=0.1):
    """
    
    Get rise time of signal, which is time from low threshold to peak.

    NOTE: Current implementation will take MAX amplitude, so even if the max
    amplitude appears later in the waveform, which will result in a large
    rise time, that is somewhat unrealistic when you look at the waveform.
    
    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.

    Returns
    -------
    rise_time : float
        Signal rise time.

    """
    peak_time = np.argmax(waveform)/10 # NOTE: time of signal peak in us 
    imin, _ = calc_signal_start_end(waveform, threshold=threshold)
    start_time = imin/10  # Note: converts index location to a start time (us)
    rise_time = peak_time - start_time
   
    return rise_time


def calc_average_freq(waveform, dt=10**-7, threshold=0.1):
    """
    
    Gets average frequency defined as the number of zero crossings
    divided by the length of the signal according to Moevus2008.

    Parameters
    ----------
    waveform : array-like
        Voltage [V] time series measurement of acoustic emission event.
    dt : float, optional
        Sample period [s]; 1/fs, fs typically is 10 MHz.
    threshold : float, optional
        Floating threshold that defines signal start/end. The default is 0.1.

    Returns
    -------
    average_frequency : float
        Average frequency of signal in Hz.

    """
    imin, imax = calc_signal_start_end(waveform, threshold=threshold)
    cut_signal = waveform[imin:imax]
    zero_crossings = calc_zero_crossings(waveform, threshold=threshold)
    average_frequency = zero_crossings/(len(cut_signal)*dt)
    
    return average_frequency 


def calc_ppmc_for_zero_shift_and_optimum_shift(s1, s2, dt=10**-7):
    """
    
    Calculates the pearson product moment correlation between two signals. 
    Ranges from -1 to 1. Value independent of signal scaling / amplitude.
    One PPMC corresponds to the signals "as is" where no shift is performed.
    The other PPMC corresponds to the signals once they're shifted to be
    at the optimum correlation.

    Parameters
    ----------
    s1 : array-like
        1D array, discrete time signal.
    s2 : array-like
        1D array, discrete time signal.

    Returns
    -------
    ppmc_no_shift : float
        Maximum pearson product moment correlation between s1 and s2.
    ppmc_shift : float
        Maximum pearson product moment correlation between s1 and shifted s2.
    time_offset : float
        The time offset between s1 and s2 maximizing correlation.
    sample_shift : int
        Number of sample offset.
    direction : str
        Direction of shift / offset.
    shifted_s2 : array-like
        Shifted s2 signal, uses zero padding. Same size as s2.

    """
    ppmc_no_shift = calc_ppmc(s1, s2)
    time_offset, sample_shift, direction, shifted_s2 = calc_time_offset(s1,s2,
                                                                        dt=dt)
    ppmc_shift = calc_ppmc(s1,shifted_s2)
    
    return ppmc_no_shift, ppmc_shift, time_offset, sample_shift, direction, \
           shifted_s2
           
    
def calc_time_offset(s1, s2, dt=10**-7):
    """
    
    Compute the time offset that maximizes the correlation between two signals.
    Also, returns shifted s2 waveform.

    Parameters
    ----------
    s1 : array-like
        Signal 1.
    s2 : arrya-like
        Signal 2. Size matches s1.
    dt : TYPE, optional
        Sampling period. The default is 10**-7.

    Returns
    -------
    time_offset : float
        The time offset between s1 and s2 maximizing correlation.
    sample_shift : int
        Number of sample offset.
    direction : str
        Direction of shift / offset.
    shifted_s2 : array-like
        Shifted s2 signal, uses zero padding. Same size as s2.

    """
    corr = signal.correlate(s1, s2)
    lags = signal.correlation_lags(len(s1), len(s2))
    corr /= np.max(corr)
    
    sample_shift = lags[np.argmax(corr)] # sample offset
    
    direction = 'left' # direction of shift of signal 2
    if sample_shift > 0: 
        direction = 'right'
    
    sample_shift = np.abs(sample_shift)
    
    print(f'Max correlation occurs when signal 2 is shifted',
          f'{direction} by {sample_shift} samples.')
    
    lags = lags*dt*10**6 # convert from samples to seconds, sec -> usec 
    
    max_corr_ind = np.argmax(corr) # index at which correlation is max
    time_offset = np.abs(lags[max_corr_ind]) 
    
    print(f'Time delay: {time_offset} microseconds going {direction}')
    
    if direction == 'left':
        shifted_s2 = np.pad(s2, (0, sample_shift), 'constant')[sample_shift:]
    elif direction == 'right':
        shifted_s2 = np.pad(s2, (sample_shift,0), 'constant')[0:len(s2)]
        
    return time_offset, sample_shift, direction, shifted_s2
    

def calc_norm_cross_corr(s1,s2):
    """
    
    Calculates the maximum normalized cross correlation between two signals. 
    Ranges from -1 to 1. Value independent of signal scaling / amplitude.
    
    Source: Digital Signal Processing: Principles, Algorithms, and Applications
            (Pg 121)

    Parameters
    ----------
    s1 : array-like
        1D array, discrete time signal.
    s2 : array-like
        1D array, discrete time signal.

    Returns
    -------
    norm_cross_corr : float
        Maximum normalized cross correlation between s1 and s2.

    """
    # Get maximum cross-correlation between s1 and s2
    rxy = np.correlate(s1,s2,'valid')
    # Get maximum auto-correlations for s1 and s2
    rxx = np.correlate(s1,s1,'valid')
    ryy = np.correlate(s2,s2,'valid')
    # Normalize 
    norm_cross_corr = rxy / (np.sqrt(rxx*ryy))
    
    return norm_cross_corr


def calc_ppmc(s1,s2):
    """
    
    Calculates the pearson product moment correlation between two signals. 
    Ranges from -1 to 1. Value independent of signal scaling / amplitude.
    If s1 and s2 are the same size. Returns only one value since there is only
    one position where they overlap. Signals are mean shifted.

    Parameters
    ----------
    s1 : array-like
        1D array, discrete time signal.
    s2 : array-like
        1D array, discrete time signal.

    Returns
    -------
    ppmc : float
        Maximum pearson product moment correlation between s1 and s2.

    """
    # Get maximum cross-correlation between s1 and s2
    rxy = np.correlate(s1-np.mean(s1),s2-np.mean(s2),'valid')
    # Get maximum auto-correlations for s1 and s2
    rxx = np.correlate(s1-np.mean(s1),s1-np.mean(s1),'valid')
    ryy = np.correlate(s2-np.mean(s2),s2-np.mean(s2),'valid')
    # Calculate pearson product moment correlation
    ppmc = rxy / (np.sqrt(rxx*ryy))
    
    return ppmc

def calc_norm_cross_corr_matrix(signals):
    """
    
    Calculates the cross-correlation matrix between signals.
    Each element (i,j) in the matrix represents the maximum normalized cross 
    correlation between signal i and signal j.
    
    Parameters
    ----------
    signals : array-like
        List of 1D arrays, each representing a discrete time signal.

    Returns
    -------
    corr_matrix : array-like
        2D array, cross-correlation matrix between signals.

    """
    # Initialize empty correlation matrix
    corr_matrix = np.empty((len(signals), len(signals)))
    # Compute normalized cross-correlation for each pair of signals
    for row in range(len(signals)):
        for col in range(len(signals)):
            corr_matrix[row, col] = calc_norm_cross_corr(signals[row],
                                                         signals[col])
                           
    return corr_matrix


def calc_avg_wave(waves):
    """
    
    Function computes average wave (separated by channel).

    Parameters
    ----------
    waves : list
        List of channels / sensors. Each index of list contains 2D array
        corresponding with shape = [# of events, sig_len]
        i.e. In the event that more than 1 channel is being used, waves[0] 
        would correspond to all the waves from sensor 0.
        
    Returns
    -------
    avgw : list / float
        Mean wave. List size depends on number of channels in 
        input variables 'waves'. Averaged over all waves from a given channel.
    stdw : list / float
        Standard deviation associated with avgw.

    """
    # Check if multiple channel
    if type(waves) == list: 
        avgw = []
        stdw = []
        for idx, channel in enumerate(waves):
            avgw.append(np.mean(channel,axis=0))
            stdw.append(np.std(channel,axis=0))
    else: # single channel
        avgw = np.mean(waves,axis=0) 
        stdw = np.mean(waves,axis=0)
        
    return avgw, stdw 


def calc_avg_norm_fft(waves, dt=10**-7, low_pass=None, high_pass=None, units='kHz'):
    """
    
    Function computes FFT for each wave passed in, in addition
    to the average / std FFTs (separated by channel).

    Parameters
    ----------
    waves: array-like
        List of voltage time series of waveforms.
    dt : TYPE, optional
        Time between samples (s) (also inverse of sampling rate). The default 
        is 10**-7.
    low_pass : int, optional
        Low pass filter threshold. The default is None.
    high_pass : int, optional
        High pass filter threshold. The default is None.
        
    Returns
    -------
    avgz : array-like
        Average amplitude.
    stdz : array-like
        Std amplitude.
    zs : array-like
        Amplitude for each waveform in waves.
    w : array-like
        Frequency.
        
    """
    zs = []
    avgz = []
    stdz = []    
    if type(waves) == list: # Multiple channels
        for sensor in waves: 
            zsensor = []
            for _,wave in enumerate(sensor):
                w,z = calc_norm_fft(wave, dt, low_pass, high_pass) 
                zsensor.append(z)
            zs.append(np.array(zsensor)) 
            avgzsensor = np.mean(zsensor, axis=0)
            stdzsensor = np.std(zsensor, axis=0)
            if units=='kHz':
                w = w/1000
            avgz.append(avgzsensor)
            stdz.append(stdzsensor)
    else: # single channel
        zsensor = []
        for _,wave in enumerate(waves):
            w,z = calc_norm_fft(wave, dt, low_pass, high_pass) 
            zsensor.append(z)
        zs.append(np.array(zsensor)) 
        avgzsensor = np.mean(zsensor,axis=0)
        stdzsensor = np.std(zsensor,axis=0)
        if units=='kHz':
            w = w/1000
        avgz.append(avgzsensor)
        stdz.append(stdzsensor)
    if len(avgz)==1: # single channel
        avgz = flatten(avgz)
        stdz = flatten(stdz)
        zs = flatten(zs)

    return avgz, stdz, zs, w


def calc_avg_fft(waves, dt=10**-7, low_pass=None, high_pass=None, units='kHz'):
    """
    
    Function computes FFT for each wave passed in, in addition
    to the average / std FFTs (separated by channel).

    Parameters
    ----------
    waves: array-like
        List of voltage time series of waveforms.
    dt : TYPE, optional
        Time between samples (s) (also inverse of sampling rate). The default 
        is 10**-7.
    low_pass : int, optional
        Low pass filter threshold. The default is None.
    high_pass : int, optional
        High pass filter threshold. The default is None.
        
    Returns
    -------
    avgz : array-like
        Average amplitude.
    stdz : array-like
        Std amplitude.
    zs : array-like
        Amplitude for each waveform in waves.
    w : array-like
        Frequency.
        
    """
    zs = []
    avgz = []
    stdz = []    
    if type(waves) == list: # Multiple channels
        for sensor in waves: 
            zsensor = []
            for _,wave in enumerate(sensor):
                w,z = calc_fft(wave, dt, low_pass, high_pass) 
                zsensor.append(z)
            zs.append(np.array(zsensor)) 
            avgzsensor = np.mean(zsensor, axis=0)
            stdzsensor = np.std(zsensor, axis=0)
            if units=='kHz':
                w = w/1000
            avgz.append(avgzsensor)
            stdz.append(stdzsensor)
    else: # single channel
        zsensor = []
        for _,wave in enumerate(waves):
            w,z = calc_fft(wave, dt, low_pass, high_pass) 
            zsensor.append(z)
        zs.append(np.array(zsensor)) 
        avgzsensor = np.mean(zsensor,axis=0)
        stdzsensor = np.std(zsensor,axis=0)
        if units=='kHz':
            w = w/1000
        avgz.append(avgzsensor)
        stdz.append(stdzsensor)
    if len(avgz)==1: # single channel
        avgz = flatten(avgz)
        stdz = flatten(stdz)
        zs = flatten(zs)

    return avgz, stdz, zs, w
