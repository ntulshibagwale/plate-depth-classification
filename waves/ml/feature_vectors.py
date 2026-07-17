import numpy as np
from scipy.integrate import simps

from waves.signal_processing import calc_fft
from waves.signal_processing import calc_freq_centroid
from waves.signal_processing import calc_average_freq
from waves.signal_processing import calc_max_amplitude
from waves.signal_processing import calc_partial_power
from waves.signal_processing import calc_peak_freq
from waves.signal_processing import calc_duration
from waves.signal_processing import calc_rise_time
from waves.signal_processing import calc_signal_energy
from waves.signal_processing import calc_zero_crossings
from waves.signal_processing import calc_decay_time
from waves.signal_processing import calc_rise_freq
from waves.signal_processing import calc_reverb_freq
from waves.signal_processing import calc_weighted_peak_freq
from waves.signal_processing import calc_wavelet_packet_decomposition
from waves.signal_processing import calc_norm_fft


def calc_alpha(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3):
    peak_freq = calc_peak_freq(waveform, dt=dt, low_pass=low_pass,
                                   high_pass=high_pass)
    max_amp = calc_max_amplitude(waveform)
        
    return [peak_freq, max_amp]


def calc_beta(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3):
    peak_freq = calc_peak_freq(waveform, dt=dt, low_pass=low_pass,
                                   high_pass=high_pass)
    freq_centroid = calc_freq_centroid(waveform, dt=dt, low_pass=low_pass,
                                   high_pass=high_pass)
        
    return [peak_freq, freq_centroid]


def calc_gamma(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3, 
               threshold=0.1):
    max_amp = calc_max_amplitude(waveform)
    rise_time = calc_rise_time(waveform, threshold=threshold)
    energy = calc_signal_energy(waveform)
    zero_crossings = calc_zero_crossings(waveform, threshold=threshold)
    
    return [max_amp, rise_time, energy, zero_crossings]


def calc_delta(waveform):
    partial_energies = calc_wavelet_packet_decomposition(waveform)
    
    return partial_energies


def calc_epsilon(waveform, dt=10**-7, threshold=0.1):
    rise_time = calc_rise_time(waveform, threshold=threshold)
    energy = calc_signal_energy(waveform)
    max_amp = calc_max_amplitude(waveform)
    duration = calc_duration(waveform, threshold=threshold)
    decay_time = calc_decay_time(waveform, threshold=threshold)
    average_freq = calc_average_freq(waveform, dt=dt, threshold=threshold)
    rise_freq = calc_rise_freq(waveform, dt=dt, threshold=threshold)
    ln_energy = np.log(energy)
    log_rt = np.log(rise_time)
    log_rd = np.log(rise_time/duration)
    log_ar = np.log(max_amp/rise_time)
    log_ad = np.log(max_amp/decay_time)
    log_af = np.log(max_amp/average_freq)

    return  [average_freq, rise_freq, log_rt, ln_energy, log_rd, log_ar, 
                 log_ad, log_af]  


def calc_zeta(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3, 
               threshold=0.1):
    rise_freq = calc_rise_freq(waveform, dt=dt, threshold=threshold)
    reverb_freq = calc_reverb_freq(waveform, dt=dt, threshold=threshold)
    average_freq= calc_average_freq(waveform, dt=dt, threshold=threshold)
    freq_centroid = calc_freq_centroid(waveform, dt=dt, low_pass=low_pass, 
                                       high_pass=high_pass)
    peak_freq = calc_peak_freq(waveform, dt=dt, low_pass=low_pass, 
                               high_pass=high_pass)
    wpf = calc_weighted_peak_freq(waveform, dt=dt, low_pass=low_pass, 
                                  high_pass=high_pass)
    pp1 = calc_partial_power(waveform, lower_bound=0, 
                             upper_bound=150*10**3)
    pp2 = calc_partial_power(waveform, lower_bound=150*10**3, 
                             upper_bound=300*10**3)
    pp3 = calc_partial_power(waveform, lower_bound=300*10**3,
                             upper_bound=450*10**3)
    pp4 = calc_partial_power(waveform, lower_bound=450*10**3, 
                             upper_bound=600*10**3)
    pp5 = calc_partial_power(waveform, lower_bound=600*10**3,
                             upper_bound=900*10**3)
    pp6 = calc_partial_power(waveform, lower_bound=900*10**3, 
                             upper_bound=1200*10**3)
    
    return [peak_freq, freq_centroid, wpf, average_freq, reverb_freq, 
                rise_freq, pp1, pp2, pp3, pp4, pp5, pp6]


def calc_eta(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3, dims=26, 
                FFT_units=1000, upsample=10001):
    feature_vector = []
    w, z = calc_fft(waveform, dt, low_pass=low_pass, high_pass=high_pass)
    w = w/FFT_units
    upsampled_w = np.linspace(low_pass, high_pass, upsample)/FFT_units 
    upsampled_z = np.interp(upsampled_w, w, z)
    dw=upsampled_w[1]-upsampled_w[0]
    interval_width = int(len(upsampled_z)/dims) 
    true_bounds = []
    for j in range(dims):
        subinterval = upsampled_z[j*interval_width: (j+1)*interval_width]
        sub_int_mass = simps(subinterval) 
        feature_vector.append(sub_int_mass) 
        true_bounds.append(low_pass/FFT_units+j*interval_width*dw)
    true_upper_bound = (j+1)*interval_width*dw+low_pass/FFT_units 
    #spacing = interval_width*dw
    if (high_pass/FFT_units-true_upper_bound)/ \
        (high_pass/FFT_units-low_pass/FFT_units)>.01:
        raise ValueError('Increase upsampling number')
        return None
    feature_vector = feature_vector/np.sum(feature_vector)
    true_bounds = np.array(true_bounds)
    feature_vector = list(feature_vector)
    
    return feature_vector

 
def calc_theta(waveform, dt=10**-7, low_pass=0, high_pass=1200*10**3):
    w, z = calc_fft(waveform, dt, low_pass=low_pass, high_pass=high_pass)
        
    return z


def calc_iota(waveform):
    
    return waveform



