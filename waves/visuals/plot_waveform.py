import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(ax, waveform, dt, sig_len, xlim_lower=0, xlim_upper=2047):
    """
    
    Plot raw event waveform with appropriate time scaling.
    
    Parameters
    ----------
    ax : matplotlib object
        Axes for plotting.
    waveform : array-like
        AE event / hit, waveform voltage values.
    dt : float
        Time between samples.
    sig_length : int
        Number of samples in waveform event / hit. 

    Returns
    -------
    ax : Matplotlib object
        Axes for plotting with waveform plotted.

    """
    duration = sig_len*dt*10**6 # convert to us
    time = np.linspace(0,duration,sig_len) # discretization of waveform time
    if type(waveform) is list:
        for idx,sig in enumerate(waveform):
            ax.plot(time,sig)
    else:
        ax.plot(time,waveform)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_xlim([time[xlim_lower],time[xlim_upper]])

    return ax, time

def plot_norm_fft(ax, fft, dt, low_pass=0, high_pass=1200*10**3):
    """
    
    Plot normalized fft.
    
    Parameters
    ----------
    ax : matplotlib object
        Axes for plotting.
    fft : array-like
        AE event / hit, normalized fft.
    dt : float
        Time between samples.

    Returns
    -------
    ax : Matplotlib object
        Axes for plotting with waveform plotted.

    """
    w = np.fft.fftfreq(len(fft), dt)
    w = w[np.where(w>=0)] 
    if low_pass is not None:
        z = z[np.where(w > low_pass)]
        w = w[np.where(w > low_pass)]
    if high_pass is not None:
        z = z[np.where(w < high_pass)]
        w = w[np.where(w < high_pass)]
        
    duration = sig_len*dt*10**6 # convert to us
    time = np.linspace(0,duration,sig_len) # discretization of waveform time
    if type(waveform) is list:
        for idx,sig in enumerate(waveform):
            ax.plot(time,sig)
    else:
        ax.plot(time,waveform)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_xlim([time[xlim_lower],time[xlim_upper]])

    return ax, time