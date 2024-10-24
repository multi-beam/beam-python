# Test construct_single_source_weights() function using random forward
# solutions, source and noise covariance matrices, and random time courses
# (time courses are only needed for evoked MER beamformer)
# This test assumes that construct_mcmv_weights() is already verified,
# and only checks that single source results match those produced
# by construct_mcmv_weights() for a single source case

import numpy as np
from numpy import dot, float_power, trace
from numpy.linalg import inv, norm
from numpy.random import randn
from construct_mcmv_weights import construct_mcmv_weights, \
        construct_single_source_weights

n_src = 5	# Number of sources
n_chan = 32	# Number of channels in the system
snr_db = -10	# Sensor level SNR in dB; -10 corresponds to amplitude SNR ~0.31
		# Two-fold change in amp SNR corresponds to ~ 6 dB
beam = "mer"	# The beamformer type: one of "mpz", "mai", "mer", "rmer"
return_h = True    # Flag to return scalar FS

# Generate random forward solutions
fs = randn(n_src, n_chan, 3)

# Generate a random noise covariance
n_cov = randn(n_chan, n_chan)
n_cov = dot(n_cov.T, n_cov)	# Ensure pos def

# For each of the sources in turn, create analytic
# covariance and calculate source orientation and
# beamformer weight

# Generate random source orientations
u_src = randn(3, n_src)

for i in range(n_src):
	u_src[:,i] = u_src[:,i]/norm(u_src[:,i])

# Generate matrix H with source FS as columns
# fs is (ns x nc x v = 3), u = (v x ns), so
# to get h_mtx = nchan x n_src we have
h_mtx = np.einsum('scv,vs->cs',fs,u_src)

# This is H*H^T - an unscaled source covariance, assuming
# that all sources are uncorrelated and have amp = 1
HHT = h_mtx @ h_mtx.T
r_cov = HHT

# Scale it to desired sensor level SNR
current_snr = trace(r_cov)/trace(n_cov)
snr_pwr = float_power(10, 0.1*snr_db)	# Power SNR in linear units
factor = snr_pwr/current_snr

# This is a full data cov with requested SNR
r_cov = factor * r_cov + n_cov	
r_inv = inv(r_cov) 	# Inverse of R

# For simplicity, we take an "evoked" value at just a single time point, with
# source time course values at this particular point completely random
s = 0.01*randn(n_src)	# Vector of source time course values at some point

# The projected to the sensors total evoked field is H*s*s^T*H^T
l = (h_mtx @ s)[:,np.newaxis]
c_avg = l @ l.T

# Get single source weights for this setting
res = construct_single_source_weights(fs, r_inv, n_cov=n_cov, beam=beam, c_avg=c_avg,
                           return_h = return_h)
w, u = res[:2]

# Get the scalar forward solutions, if requested, and verify
# that those are correct. NOTE that they won't match h_mtx,
# because single source orientations are not accurate
if return_h:
    h = res[2]
    assert np.allclose(h, np.einsum('scv,vs->cs',fs,u))

# Verify that weights and sources match those obtained using
# the construct_mcmv_weights() for a single source

for i in range(n_src):
    res = construct_mcmv_weights(fs[i,:,:][np.newaxis,:,:],
                    r_inv, n_cov = n_cov, beam = beam, c_avg = c_avg,
                    return_h = return_h)
    wi, ui = res[:2]
    assert np.allclose(w[:,i],wi[:,0])
    assert np.allclose(u[:,i],ui[:,0])

    # Get the scalar forward solutions, if requested
    if return_h:
        hi = res[2]
        assert np.allclose(h[:,i], hi[:,0])

print("\nVerify that single source results match multi-source for a single source case - PASSED.\n")
print(f'SNR = {snr_db} dB')

# Verification. Print the true source orientations:
print("True source orientations:\n", u_src)

# Print the beamformer-found source orientations:
print("Single source source orienatations (ok not to match the true):\n", u)



