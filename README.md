# Identifying Acoustic Emission Depth in Plate-like Structures with Deep Learning

This repository contains the code accompanying the paper

> **Identifying Acoustic Emission Depth in Plate-like Structures with Deep Learning**

## Feature Representations

The machine learning models were evaluated using nine acoustic emission feature representations of increasing complexity, ranging from simple handcrafted descriptors to the raw waveform.

| Feature | Length | Description | Reference |
|:-------:|------:|-------------|:---------:|
| **α** | 2 | Peak frequency, Maximum amplitude | [28] |
| **β** | 2 | Peak frequency, Frequency centroid | [29] |
| **γ** | 4 | Maximum amplitude, Rise time, Energy, Zero crossings | [5] |
| **δ** | 8 | Wavelet packet energy distribution | [30] |
| **ε** | 8 | Average frequency, Rise frequency, ln(rise time), ln(energy), ln(rise time/duration), ln(maximum amplitude/rise time), ln(maximum amplitude/decay time), ln(maximum amplitude/average frequency) | [16] |
| **ζ** | 12 | Peak frequency, Frequency centroid, Weighted peak frequency, Average frequency, Reverb frequency, Rise frequency, Partial power (0–150 kHz), (150–300 kHz), (300–450 kHz), (450–600 kHz), (600–900 kHz), (900–1200 kHz) | [14] |
| **η** | 26 | Partial power spectrum (0–1200 kHz, 46.1 kHz intervals) | [14] |
| **θ** | 245 | Fast Fourier Transform (FFT) magnitude spectrum | — |
| **ι** | 2048 | Raw acoustic emission waveform | — |

## Models

### Classical Machine Learning

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

### Neural Networks

- Fully Connected Neural Networks
- One-dimensional Convolutional Neural Networks (CNNs)

## Contact

**Nick Tulshibagwale**

Mechanical Engineering, University of California, Santa Barbara

📧 ntulshibagwale@ucsb.edu
