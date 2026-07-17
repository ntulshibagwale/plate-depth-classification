import os
import sys
import matplotlib.pyplot as plt

# Boiler plate code to import waves module
waves_dir = r'..\waves'
print(f"Waves module stored at: {waves_dir}")
if sys.path[0] != os.path.dirname(waves_dir):
    sys.path.insert(0, os.path.dirname(waves_dir))
print(f"Directory added to path: {os.path.dirname(waves_dir)}")
print("Waves modules can be imported.")

from waves.ml.feature_vectors import calc_alpha
from waves.ml.feature_vectors import calc_beta
from waves.ml.feature_vectors import calc_gamma
from waves.ml.feature_vectors import calc_delta
from waves.ml.feature_vectors import calc_epsilon
from waves.ml.feature_vectors import calc_zeta
from waves.ml.feature_vectors import calc_eta
from waves.ml.feature_vectors import calc_theta
from waves.ml.feature_vectors import calc_iota
from waves.load_data import load_json_file_from_path

DATA_PATH = r'G:\My Drive\phd\projects\coupling\ml\coupling\stratified\01\01_test.json'
coupling1_data = load_json_file_from_path(DATA_PATH)

waveform = coupling1_data['waves'][0]

alpha = calc_alpha(waveform)
beta = calc_beta(waveform)
gamma = calc_gamma(waveform)
delta = calc_delta(waveform)
epsilon = calc_epsilon(waveform)
zeta = calc_zeta(waveform)
eta = calc_eta(waveform)
theta = calc_theta(waveform)
iota = calc_iota(waveform)

print(f'alpha length: {len(alpha)}')
print(f'beta length: {len(beta)}')
print(f'gamma length: {len(gamma)}')
print(f'delta length: {len(delta)}')
print(f'epsilon length: {len(epsilon)}')
print(f'zeta length: {len(zeta)}')
print(f'eta length: {len(eta)}')
print(f'theta length: {len(theta)}')
print(f'iota length: {len(iota)}')

from sklearn import preprocessing
import numpy as np
x=[]
for index, wave in enumerate(coupling1_data['waves']):
    x.append(calc_beta(wave))

x = np.array(x)
scaler = preprocessing.StandardScaler().fit(x) 
# else: it was previously fit already
x_scaled = scaler.transform(x)  