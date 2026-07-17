This repository contains the code accompanying the paper

**[Identifying Acoustic Emission Depth in Plate-like Structures](https://link.springer.com/article/10.1007/s40192-026-00470-5)**

### Experimental Datasets

The models were trained and evaluated using two benchmark datasets consisting of acoustic emission (AE) waveforms generated from pencil-lead break (PLB) experiments on an aluminum plate. Each waveform belongs to one of three classes corresponding to the source depth:

- **Top**
- **Side**
- **Bottom**

Two datasets were constructed to evaluate model classification under realistic experimental variations. Each dataset contains 1,680 labeled waveforms (560 per class).

- **Coupling Dataset** – evaluates generalization across repeated sensor mounting and remounting (different coupling conditions).
- **Distance Dataset** – evaluates generalization across unseen source-to-sensor distances by varying the PLB location along the plate.

### Feature Extraction

Each waveform was represented using one of nine feature vectors. 

| Feature | Length | Description |
|:-------:|------:|-------------|
| **α** | 2 | Peak frequency, Maximum amplitude
| **β** | 2 | Peak frequency, Frequency centroid 
| **γ** | 4 | Maximum amplitude, Rise time, Energy, Zero crossings
| **δ** | 8 | Wavelet packet energy distribution
| **ε** | 8 | Average frequency, Rise frequency, ln(rise time), ln(energy), ln(rise time/duration), ln(maximum amplitude/rise time), ln(maximum amplitude/decay time), ln(maximum amplitude/average frequency) 
| **ζ** | 12 | Peak frequency, Frequency centroid, Weighted peak frequency, Average frequency, Reverb frequency, Rise frequency, Partial power (0–150 kHz), (150–300 kHz), (300–450 kHz), (450–600 kHz), (600–900 kHz), (900–1200 kHz) 
| **η** | 26 | Partial power spectrum (0–1200 kHz, 46.1kH z intervals) 
| **θ** | 245 | Fast Fourier Transform (FFT) magnitude spectrum 
| **ι** | 2048 | Raw acoustic emission waveform 

### Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Fully Connected Neural Networks
- One-Dimensional Convolutional Neural Networks (CNNs)

## Repository Organization

The `run_ml_experiments.py` script is the primary entry point for training and evaluating the machine learning and deep learning models described in the paper.

The repository is organized as follows:

- `load_in_feature_vectors.py` – Helper functions for extracting features from acoustic emission waveforms.
- `waves/` – Core package containing utilities for feature extraction, data loading, model definitions, training, evaluation, visualization, and Weights & Biases integration.
- `coupling/coupling_individual/` – Original AE waveforms grouped by individual sensor coupling condition.
- `distance/distance_individual/` – Original AE waveforms grouped by individual source-to-sensor distance.
- `coupling/stratified/` – Coupling dataset reorganized into the training and testing splits used for 10-fold cross-validation.
- `distance/stratified/` – Distance dataset reorganized into the training and testing splits used for 10-fold cross-validation.

For each fold, waveforms from nine experimental conditions are combined to form the training set, while the remaining condition is held out as the test set. This process is repeated until every coupling condition or source location has served as the test set exactly once. By splitting the data according to experimental condition rather than randomly sampling individual waveforms, the evaluation measures each model's ability to generalize to previously unseen coupling conditions or source-to-sensor distances.

### Contact

If you have any questions about the repository, implementation, or dataset organization, please feel free to reach out.

**Nick Tulshibagwale**  
University of California, Santa Barbara  
📧 ntulshibagwale@ucsb.edu
