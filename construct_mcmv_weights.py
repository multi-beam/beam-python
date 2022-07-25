# -------------------------------------------------------------------------
# construct_mcmv_weights using K-matrix eigenvectors parsing
# A.Moiseev, BCNI, Jul 2021
# -------------------------------------------------------------------------

import numpy as np
#import matplotlib.pyplot as plt		# For DEBUG only
from numpy import dot, trace
from numpy.linalg import cholesky, eig, inv, norm

# -------------------------------------------------------------------------
# Construct scalar MCMV beamformer weights for a fixed set of n_src dipole locations,
# given:
#	- a triplet of forward solutions (FS) for each location (namely, FS
# 	  for dipoles oriented along x, y, z directions in the head coordinates),
# 	- an inverse of full data covariance matrix,
#	- a noise covariance matrix, 
# and for the evoked beamformers:
#	- evoked fields second moments matrix.
#
# Input:
#	fs	(n_src x n_chan x 3) Numpy array of "vector" FS for the source
#		voxels. For example, fs[i,:,1] is a forward solution for a dipole
#		located in the i-th source voxel and oriented along Y axis. When
#		"vector" FS are provided, source orientations will be calculated
#		by parsing eigenvectors of the K-matrix.
#				OR
#		(n_chan x n_src) Numpy array of "scalar" FS for the sources.
#		Each column of FS then corresponds to lead field of a dipole
#		with known orientation.
#		!!! When scalar FS are provided, MSMV calculation of source
#		!!! orientations is skipped and the weights W
#		!!! 	W = R^-1 H (H^T R^-1 H)^-1
#		!!! are returned right away.
#	r_inv	(n_chan x n_chan) An inverse of the data covariance matrix.
#	n_cov	(n_chan x n_chan). The noise covariance matrix. If not specified
#		(n_cov = None, a bad idea), a diagonal white noise will be used
#	beam	(string) The beamformer type: one of "mpz", "mai", "mer", "rmer"
#	c_avg	(n_chan x n_chan) Matrix of the second moments of the trial-
#		averaged epochs, typically additionally averaged over a 
#		time interval of interest:
#			c_avg = < e(t) e^T(t) >
#		where e(t) is a column vector of trial-averaged sensor 
#		readings at time "t", and brackets <...> denote averaging
#		over a chosen time interval
#
# Output:	(w_mtx, u), where:
#	w_mtx 	(n_chan x n_src) numpy array of the beamformer weights
#	u	(3 x n_src)	array of source orientations, or None
#		if scalar FS were provided.
# -------------------------------------------------------------------------
def construct_mcmv_weights(fs, r_inv, n_cov = None, beam = "mpz", c_avg = None):
# -------------------------------------------------------------------------
	# The dynamic range of eigenvalues is affected by the dynamic range
	# of the EVs of the source correlation matrix. To get an idea of the
	# latter, note that for all equal mutual correlations there will be
	# a single big eigenvalue E and all other small eigenvalues e = 1-cc.
	# To find E write the trace: trace(corr) = n_src = E + (n_src -1)(1-cc)
	# Thus E = n_src -(n_src -1)(1-cc) and
	#	e/E = (1-cc)/[n_src -(n_src -1)(1-cc)]
	# For large enough n_src we get e/E ~ (1/n_src) * (1-cc)/cc
	# So for realistic cc the dynamic range of EVs of corr mtx
	# is ~ 1/n_src, but for 1-cc << 1 (i.e. 0.01 for 99% correlations)
	# it may reach high values (~10^-3)
	MIN_RELATIVE_E_VAL = 0.001	# Min allowed ratio of the smallest (e_val -1) to the max (e_val-1)
					# for power localizers, or without subtracting 1 for evoked localizers
	beam_types = {"mpz":k_mpz, "mai":k_mai, "mer":k_mer, "rmer":k_rmer}

	if len(fs.shape) == 3:
		# Vector forward solutions - calculate orientations
	
		n_src, n_chan, tmp = fs.shape

		# Sanity checks
		if tmp != 3:
			raise ValueError("fs should be a (n_src x n_chan x 3) numpy array")

		if r_inv.shape != (n_chan,n_chan):
			raise ValueError("r_inv should be a ({} x {}) numpy array".
					format(n_chan, n_chan))

		if not is_pos_def(r_inv):
			raise ValueError("r_inv should be a symmetric positively defined matrix")

		# Create a diagonal noise covariance matrix with a trace
		# = 0.01/trace(r_inv) (kind-of 10 dB SNR)
		if n_cov is None:
			n_cov = 0.01 * trace(r_inv) * np.identity(n_chan)
		elif n_cov.shape != (n_chan,n_chan):
			raise ValueError("n_cov should be a ({} x {}) numpy array".
					format(n_chan, n_chan))
		elif not is_pos_def(n_cov):
			raise ValueError("n_cov should be a symmetric positively defined matrix")

		n_inv = inv(n_cov)
		beam = beam.lower()

		if not (beam in beam_types):
			raise ValueError("beam should be one of: {}".format([*beam_types]))

		is_evoked = False	# Flag that this is an evoked beamformer	

		if (beam in ("mer", "rmer")):
			if c_avg is None:
				raise ValueError("An evoked beamformer '{}' is requested, but c_avg is not specified"
						.format(beam))
			elif c_avg.shape != (n_chan,n_chan):
				raise ValueError("c_avg should be a ({} x {}) numpy array".
					format(n_chan, n_chan))

			is_evoked = True

		# L-matrix is simply a concatenated (along the voxels) version of fs:
		l_mtx = np.concatenate(fs, axis = 1)	# this is a (n_chan x (n_src*3)) matrix

		# Construct the K-matrix (in MCBF paper terms) depending on the beamformer type:
		k_mtx = beam_types[beam](l_mtx, r_inv, n_cov, n_inv, c_avg)

		# Perform K-matrix EVD; we only need at most 1st n_src biggest eigenvectors
		e_vals, e_vecs = eig(k_mtx)

		# NOTE: Due to rounding errors the smallest 2*n_src e_vals, e_vecs have infinitasimal
		# complex parts, causing the whole output to be cast to complex domain.
		# Therefore:
		e_vals = np.real(e_vals)
		e_vecs = np.real(e_vecs)

		# DEBUG: yes, this might happen, but only due to rounding errors in last decimal digits
		# if (not is_evoked) and np.any(e_vals < 1.):
		#	print("WARNING: K-matrix eigenvalue smaller than 1 found")

		idx = np.argsort(-e_vals)[:n_src]	# Indecies of n_src largest e_vals
		e_vals = e_vals[idx]
		e_vecs = e_vecs[:,idx]

		# Discard eigenvectors with e_vals too close to 1 or 0
		tmp = 0. if is_evoked else 1.	# Value to subtract from e_vals depending on beam type 
		tmp = (e_vals - tmp)
		idx = (tmp / tmp[0] >= MIN_RELATIVE_E_VAL) # Keep only large enough eigenvalues
		e_vals = e_vals[idx]	# Actually, those are never used
		e_vecs = e_vecs[:,idx]

		if np.any(np.iscomplex(e_vecs)):
			raise ValueError("Got complex-valued eigenvectors of the K-matrix")

		# Get source orientations
		u = parse_eigenvectors(e_vecs)

		# Get scalar forward solutions
		h_mtx = np.zeros((n_chan, n_src))
		for i_src in range(n_src):
			h_mtx[:,i_src] = dot(fs[i_src], u[:,i_src])
	else:
		# fs are already scalar forward solutions
		h_mtx = fs
		u = None

	# Calculate the weights according to the expression
	# W = R^-1 H (H^T R^-1 H)^-1
	s_inv = inv(dot(h_mtx.T, dot(r_inv, h_mtx)))	# Inverse of S-matrix
	w_mtx = dot(r_inv, dot(h_mtx, s_inv))

	return (w_mtx, u)

# -------------------------------------------------------------------------
# Check if an array is a symmetric positively defined matrix
# Returns True / False
# -------------------------------------------------------------------------
def is_pos_def(a):
# -------------------------------------------------------------------------
    try:
        cholesky(a)
        return True
    except:
        return False

# K-matrix calculation for MPZ: K = T^-1 S, where
#	S = L^T R^-1 L
#	T = L^T R^-1 N R^-1 L
# (n_inv, c_avg are not used)
#-----------------------------------------
def k_mpz(l_mtx, r_inv, n_cov, n_inv, c_avg):
#-----------------------------------------
	lt_rm1 = dot(l_mtx.T, r_inv)
	s = dot(lt_rm1, l_mtx)
	t = dot(dot(lt_rm1, n_cov), lt_rm1.T)
	return dot(inv(t), s)

# K-matrix calculation for MAI: K = S^-1 G, where
#	S = L^T R^-1 L
#	G = L^T N^-1 L
# (n_cov, c_avg are not used)
#-----------------------------------------
def k_mai(l_mtx, r_inv, n_cov, n_inv, c_avg):
#-----------------------------------------
	lt_rm1 = dot(l_mtx.T, r_inv)
	lt_rn1 = dot(l_mtx.T, n_inv)
	s = dot(lt_rm1, l_mtx)
	g = dot(lt_rn1, l_mtx)
	return dot(inv(s), g)

# K-matrix calculation for MER: K = T^-1 E, where
#	T = L^T R^-1 N R^-1 L
#	E = L^T R^-1 Cavg R^-1 L
# (n_inv is not used)
#-----------------------------------------
def k_mer(l_mtx, r_inv, n_cov, n_inv, c_avg):
#-----------------------------------------
	lt_rm1 = dot(l_mtx.T, r_inv)
	t = dot(dot(lt_rm1, n_cov), lt_rm1.T)
	e = dot(dot(lt_rm1, c_avg), lt_rm1.T)
	return dot(inv(t), e)

# K-matrix calculation for rMER: K = S^-1 E, where
#	S = L^T R^-1 L
#	E = L^T R^-1 Cavg R^-1 L
# (n, n_inv are not used)
#-----------------------------------------
def k_rmer(l_mtx, r_inv, n_cov, n_inv, c_avg):
#-----------------------------------------
	lt_rm1 = dot(l_mtx.T, r_inv)
	s = dot(lt_rm1, l_mtx)
	e = dot(dot(lt_rm1, c_avg), lt_rm1.T)
	return dot(inv(s), e)

# Parse eigenvectors of the K-matrix and find source
# orientations
# Each EV is considered as concatenation of n_src 3D vectors, each
# one being parallel to the orientation vector of a corresponding source.
# Thus in ideal situations k-th triplets of each eigenvector should be
# parallel to each other. In real life this is not so. Therefore we
# determine k-th source orienation as an average of all k-th
# triplets orientations weighted by their powers (assuming that EVs
# of the K-matrix are all normalized to 1)
#
# Input:
#	v	((3 n_src) x (n_vec)) matrix with columns which are
#		n_vec largest eigenvectors of the K-matrix
# Output:
#	u	(3 x n_src) matrix of source orientations (as columns)
#------------------------
def parse_eigenvectors(v):
#------------------------
	# Calculate powers of each source orientation in each EV
	n_src = int(v.shape[0]/3)
	n_vec = v.shape[1]

	powers = np.zeros((n_src, n_vec))
	for i_src in range(n_src):
		i1 = i_src * 3
		i2 = i1 + 3

		for i_vec in range(n_vec):
			a = v[i1:i2, i_vec]
			powers[i_src, i_vec] = dot(a, a)

	# For each source, find a vector # with the maximum power
	idx_vec = np.argmax(powers, axis = 1)	# 1D list of n_src values of vec ##

	# Find the weighted average orientations
	u = np.zeros((3, n_src))

	for i_src in range(n_src):
		i1 = i_src * 3
		i2 = i1 + 3
		
		# Choose the max power EV to define orientation sign
		t = v[i1:i2, idx_vec[i_src]]

		for i_vec in range(n_vec):
			a = v[i1:i2, i_vec]

			if dot(a, t) > 0:
				u[:,i_src] = u[:,i_src] + powers[i_src, i_vec]*a
			else:	# Flip the sign of 'a'
				u[:,i_src] = u[:,i_src] - powers[i_src, i_vec]*a

		# Normalize the orientation
		u[:,i_src] = u[:,i_src]/norm(u[:,i_src])

	return u

# -------------------------------------------------------------------------
# Construct single source beamformer weights for a fixed set of n_src dipole locations,
# given:
#	- a triplet of forward solutions (FS) for each location (namely, FS
# 	  for dipoles oriented along x, y, z directions in the head coordinates),
# 	- an inverse of full data covariance matrix,
#	- a noise covariance matrix, 
# and for the evoked beamformers:
#	- evoked fields second moments matrix.
#
# This function has the exact same signature as construct_mcmv_weights() and internally
# calls the latter for each source in turn.
#
# Input:
#	fs	(n_src x n_chan x 3) Numpy array of "vector" FS for the source
#		voxels. For example, fs[i,:,1] is a forward solution for a dipole
#		located in the i-th source voxel and oriented along Y axis. When
#		"vector" FS are provided, source orientations will be calculated
#		by parsing eigenvectors of the K-matrix.
#				OR
#		(n_chan x n_src) Numpy array of "scalar" FS for the sources.
#		Each column of FS then corresponds to lead field of a dipole
#		with known orientation.
#		!!! When scalar FS are provided, MSMV calculation of source
#		!!! orientations is skipped and the weights W
#		!!! 	W = R^-1 H (H^T R^-1 H)^-1
#		!!! are returned right away.
#	r_inv	(n_chan x n_chan) An inverse of the data covariance matrix.
#	n_cov	(n_chan x n_chan). The noise covariance matrix. If not specified
#		(n_cov = None, a bad idea), a diagonal white noise will be used
#	beam	(string) The beamformer type: one of "mpz", "mai", "mer", "rmer"
#	c_avg	(n_chan x n_chan) Matrix of the second moments of the trial-
#		averaged epochs, typically additionally averaged over a 
#		time interval of interest:
#			c_avg = < e(t) e^T(t) >
#		where e(t) is a column vector of trial-averaged sensor 
#		readings at time "t", and brackets <...> denote averaging
#		over a chosen time interval
#
# Output:	(w_mtx, u), where:
#	w_mtx 	(n_chan x n_src) numpy array of the beamformer weights
#	u	(3 x n_src)	array of source orientations, or None
#		if scalar FS were provided.
# -------------------------------------------------------------------------
def construct_single_source_weights(fs, r_inv, n_cov = None, beam = "mpz", c_avg = None):
# -------------------------------------------------------------------------
	if len(fs.shape) == 3:
		# Vector forward solutions; need to calculate orientations	
		n_src, n_chan, _ = fs.shape
		w_mtx = np.zeros((n_chan, n_src))
		u = np.zeros((3, n_src))

		for i in range(n_src):
			f = fs[i,:,:][np.newaxis,:]	# Make the shape (1, n_chan, 3) 
			# Returned w, u1 shapes are (n_chan x 1), (3 x 1)
			w, u1 = construct_mcmv_weights(f, r_inv, n_cov, beam, c_avg)
			w_mtx[:,i] = w[:,0]
			u[:,i] = u1[:,0]
	else:
		# Scalar forward solutions
		n_chan, n_src = fs.shape
		w_mtx = np.zeros((n_chan, n_src))
		u = None

		for i in range(n_src):
			f = fs[:,i]
			# Returned w's shape is (n_chan x 1)
			w, _ = construct_mcmv_weights(f[:, np.newaxis], r_inv, n_cov, beam, c_avg)
			w_mtx[:,i] = w[:,0]

	return (w_mtx, u)

