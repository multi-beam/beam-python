# beam-python
The main routine that finds source orientations and calculates MCMV beamformer weights
for a fixed set of locations is ***construct\_mcmv\_weights.py***. The calling parameters
and return values are documented in comments at the top of the file.

The ***test\_mcmv\_weights.py*** is a heavily commented example script which
demonstrates precise reconstruction of source orientations with randomly generated
forward solutions and a random noise covariance matrix, irrespective of the source SNR. Precise
reconstruction is possible because exact analytical expression for the full data covariance
matrix is supplied to the ***construct\_mcmv\_weights()*** function.

