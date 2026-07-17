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

### Model Evaluation

Performance was evaluated using **10-fold cross-validation**, where each fold represented an unseen experimental condition rather than a random subset of waveforms. Model performance was quantified using classification accuracy on the held-out fold. 

## Repository Organization and Model Evaluation

The `run_ml_experiments.py` script is used to train and evaluate the different models.
The dataset folders are organized as follows:

- `coupling/coupling_individual/` contains the AE waveforms collected under each individual sensor-coupling condition.
- `distance/distance_individual/` contains the AE waveforms collected at each individual source-to-sensor distance.
- `coupling/coupling/stratified/` contains the same coupling data reorganized into training and testing splits.
- `distance/distance/stratified/` contains the same distance data reorganized into training and testing splits.

For each stratified fold, waveforms from nine experimental conditions are combined into the training set, while the remaining condition is reserved as the test set. This process is repeated so that every coupling condition or source location serves as the held-out test condition once. This organization evaluates whether the models can generalize to an experimental condition that was not represented in the training data, rather than only evaluating performance on randomly selected waveforms from previously observed conditions.

Questions about the repository are welcome if any part of the code or dataset organization is unclear.

**Nick Tulshibagwale**

ntulshibagwale@ucsb.edu
