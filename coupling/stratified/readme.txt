Using 9 out of the 10 coupling datasets for training,
the remaining coupling dataset will be used as test.

Ex:
01 -> coupling01 is 01_test.json and coupling02-coupling10 is used for train, 02_train.json
06 -> coupling06 is 06_test.json etc...

Each coupling dataset contains 168 waveforms, 56 per orientation

Train will always have 168 * 9 = 1512 waveforms (504 per orientation)
Test will always have 168 * 1 = 168 waveforms (56 per orientation)