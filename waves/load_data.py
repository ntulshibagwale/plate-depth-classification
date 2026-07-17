import json
import numpy as np


def read_ae_file(ae_file):
    """
    
    Function loads in experimental AE data from .txt files generated from
    Digital Wave software. 
    
    Parameters
    ----------
    ae_file : str
        File path of .txt file containing all waveforms (voltage over time).
    
    Returns
    -------
    signals : array-like
        Each index of list points to list of waveform signals from a sensor.
        i.e signals[0] will return all the AE hits from sensor 1, and 
        signals[0][0] will return the 1st waveform from sensor 1 (with size
        sig_length).
    ev : array-like
        List of event numbers, indexed from 1 (first event in test). All 
        sensors trigger at same time, so events are equivalent in time.
    fs : int
        Sampling frequency. Typically 10**7 Hz or 10 MHz with Digital Wave.
    channel_num : int
        Number of channels / AE sensors used.
    sig_length : int
        Number of samples in waveform event / hit. 

    """
    # Read in .txt file generated from Digital Wave DAQ / Software
    f = open(ae_file)
    data = f.readlines()
    f.close()
    
    # Get the signal processing parameters from header
    header = data[0]
    fs = int(header.split()[0]) * 10**6  # DAQ sampling freq (usually 10 MHz)
    sig_length = int(header.split()[2])  # Number of samples in waveform event
    channel_num = int(header.split()[3]) # Number of AE sensors used
    
    # Read in waveform data and turn into list of sensors pointing to AE hits
    lines = data[1:]
    signals = [] 
    
    # Loop through the columns taken from .txt file (waves from each sensor)
    for channel in range(0,channel_num):
        # Get data from the sensor's respective column
        v = np.array([float(line.split()[channel]) for line in lines])
        # Turn the long appended column into separate AE hits using sample num
        z = []
        for i in range(0,len(v),sig_length):
            z.append(v[i:i+sig_length])    
        signals.append(z)
    
    # Create array of corresponding event numbers 
    ev = np.arange(len(signals[0]))+1 # all sensors have same number of events

    return signals, ev, fs, channel_num, sig_length


def load_scenario(data, loc, num_sensors=2):
    """
    
    Load in waves from a given plb orientatin.

    Parameters
    ----------
    data : dict
        AE experimental data, loaded from .json file.
    loc : str
        Source type.
    num_sensors : int, optional
        Number of sensors to load in. If user inputs 2, will pull data from
        sensor 1 and sensor 2. Default is 2.

    Returns
    -------
    waves_from_scenario : array-like
        Waveforms from experiment. Index 0 contains waves from sensor 1, 
        index 1 contains waves from sensor 2, ... etc.
    events_from_scenario : array-like
        List of event # from experiment.

    """
    waves = data['waves']
    event = data['event']
    
    waves_from_scenario = []
    for sensor in range(num_sensors):
        waves_from_scenario.append(waves[
                       (data['sensor'] == sensor+1) 
                       & (data['location'] == loc)])
        
        
    # Get the recorded event / ae hit numbers 
    events_from_scenario = event[
        (data['sensor'] == 1)
        & (data['location'] == loc)] 
    
    print(f"# of {loc} channel waves : ",
          f"{len(events_from_scenario)}")
    
    return waves_from_scenario, events_from_scenario


def load_json_file_from_path(path):
    """
    
    Loads in .json file from file path.

    Parameters
    ----------
    path : str
        Path name the .json file is located at. 

    Returns
    -------
    data : dict
        Returns dict stored in .json file.

    """
    print(f"Loading in Dataset from {path}")
    with open(path) as json_file:
        data = json.load(json_file)
    for key in data.keys():
        data[key]  = np.array(data[key])
    print("Successfully loaded in .json file from path.\n")
    
    return data

    
