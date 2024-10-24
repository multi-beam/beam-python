# Test MCMV weights calculations with random forward solutions,
# source and noise covariance matrices, and random time courses
# (time courses are only needed for evoked MER beamformer)

import numpy as np
from numpy import dot, float_power, trace
from numpy.linalg import inv, norm
from numpy.random import randn
from construct_mcmv_weights import construct_mcmv_weights

n_src = 5	# Number of sources
n_chan = 32	# Number of channels in the system
snr_db = -20	# Sensor level SNR in dB; -10 corresponds to amplitude SNR ~0.31
		# Two-fold change in amp SNR corresponds to ~ 6 dB
cc = 0.9	# Mutual correlation coefficient between all sources
beam = "mpz"	# The beamformer type: one of "mpz", "mai", "mer", "rmer"
return_h = True    # Flag to return scalar FS

# Generate random forward solutions
fs = randn(n_src, n_chan, 3)

# Generate a random noise covariance
n_cov = randn(n_chan, n_chan)
n_cov = dot(n_cov.T, n_cov)	# Ensure pos def

# Generate random source orientations
u_src = randn(3, n_src)
for i in range(n_src):
	u_src[:,i] = u_src[:,i]/norm(u_src[:,i])

# Generate an analytic full covariance matrix R with a specified source SNR
# This is the source correlation matrix:
corr_mtx = cc*np.ones((n_src,n_src)) + (1 - cc)*np.identity(n_src)
snr_pwr = float_power(10, 0.1*snr_db)	# Power SNR in linear units

# The expression is R = H*C*H^T + N, where C is the source covariance
# matrix. We assume that all sources have equal amplitudes, and that the 
# sensor level SNR is a ratio of a trace of the source part of full cov
# to the trace of the noise part of full cov

# Generate matrix H with source FS as columns
h_mtx = np.zeros((n_chan, n_src))
for i in range(n_src):
	h_mtx[:,i] = dot(fs[i], u_src[:,i])

# This is H*C*H^T - an unscaled source covariance 
r_cov = dot(h_mtx, dot(corr_mtx, h_mtx.T))

# Scale it to desired sensor level SNR
current_snr = trace(r_cov)/trace(n_cov)
factor = snr_pwr/current_snr

# This is a full data cov with requested SNR
r_cov = factor * r_cov + n_cov	
r_inv = inv(r_cov) 	# Inverse of R

# For simplicity, we take an "evoked" value at just a single time point, with
# source time course values at this particular point completely random
s = 0.01*randn(n_src)	# Vector of source time course values at some point

# The projected to the sensors total evoked field is H*s*s^T*H^T
l = np.reshape(dot(h_mtx, s), (n_chan,1))	# Make it a column vector

# This is Cavg matrix needed for evoked beamformers (mer, rmer)
c_avg = dot(l, l.T)

# Get the weights
res = construct_mcmv_weights(fs, r_inv, n_cov = n_cov, beam = beam, c_avg = c_avg,
                                return_h = return_h)
w, u = res[:2]

# Get the scalar forward solutions, if requested
if return_h:
    h = res[2]
    assert np.allclose(h, np.einsum('scv,vs->cs',fs,u))

# Verification. Print the true source orientations:
print("u_src:\n", u_src)

# Print the beamformer-found source orientations:
print("u:\n", u)

# Print the weight matrix W:
print("\nw_mtx:\n", w)

