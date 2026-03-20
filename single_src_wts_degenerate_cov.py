# This example demonstrates calculation of single source beamformer
# weights in case of a degenerate data covariance matrix. 
# For generality, random forward solutions and random time courses
# are used.

import numpy as np
from numpy import dot, float_power, trace
from numpy.linalg import pinv, norm, matrix_rank
from numpy.random import randn
from construct_mcmv_weights import get_default_noise_covariance, \
        construct_single_source_weights, is_pos_def

def main():
    # ---------- Settings ------------------------------
    n_src = 5	    # Number of sources/number of weights to construct
    n_chan = 32	    # Number of sensor channels in the system
    data_rank = 28  # Number of actually independent channels (i.e. the data rank)
    beam = "mpz"	# The beamformer type; do not use evoked beams in this example
    lmbda = 1e-8    # Regularization parameter used for degenerate covs
    return_h = True # Flag to return scalar FS

    snr_db = -10	# Sensor level SNR in dB for simulated data; -10 corresponds
                    # to amplitude SNR ~0.31
    seed = 42       # Seed for random generator
    # ---------- End of settings -----------------------

    rng = np.random.default_rng(seed=seed)      # Setup random gen for reproducability
    data_rank = max(min(data_rank, n_chan),1)

    '''
    ---------------------------------------------------------------------
    Here we construct our example surrogate test data: possibly degenerate
    covariance matrices and forward solutions.

    You can replace the call to construct_test_data() with loading your own
    data. To calculate the beamformer weights, you will need to supply: an
    array of "vector" forward solutions for n_src locations of interest (fs);
    the inverse of the full data covariance matrix or its regularized pseudo-
    inverse (r_inv); the noise covariance matrix (n_cov) or its regularized
    version for degenerate data. In practice you may not be able to calculate
    the n_cov from your data - then a substitute should be provided, such as
    n_diag below.

    The construct_test_data() function does return the true noise covariance
    n_cov and also true source orientations u_src. Those are never available
    in practice and are only used for comparisons with the beamformer results.
    ---------------------------------------------------------------------
    '''

    # Below:
    #   fs (n_src,n_chan,3) - array of "vector" forward solutions
    #       for sources;
    #   u_src (3, n_src) - true source orientations;
    #   h_mtx (n_chan, n_src) - array of "scalar" forward solutions
    #       for sources;
    #   n_cov(n_chan, n_chan) - the true (unknown) noise covariance matrix
    #   r_cov(n_chan, n_chan) - the "measured" full covariance matrix
    fs, u_src, h_mtx, n_cov, r_cov = construct_test_data(
        n_src=n_src,
        n_chan=n_chan,
        data_rank=data_rank,
        snr_db=snr_db,
        rng=rng,
    )

    '''
    -------------------------------------------------------------------
    After loading the data, one needs to calculate the inverse of r_cov
    -------------------------------------------------------------------
    '''
    # Use pseudo-inverse to invert full data cov in any case - this is safe 
    r_inv = pinv(r_cov)

    '''
    --------------------------------------------------------------------
    For rank-deficient data, one needs to regularize all covariances used
    in weights calculations. We first assume that true n_cov is available.
    --------------------------------------------------------------------
    '''
    if data_rank < n_chan:
        regularize = lambda a: a + (lmbda*np.trace(a)/a.shape[0])*np.eye(n_chan)

        r_inv = regularize(r_inv)
        n_cov = regularize(n_cov)
        r_cov = regularize(r_cov)

    '''
    --------------------------------------------------------------------------
    Now, find single source weights w using the TRUE noise covariance. Note that
    when n_src > 1, we may not get precise orientations because strictly speaking
    multi-source weights should be calculated in this case. If n_src = 1 we should
    obtain exact match with the true orientations for the surrogate data (up to a
    sign and rounding noise) - irrespective to SNR.
    --------------------------------------------------------------------------
    '''
    res = construct_single_source_weights(fs, r_inv, n_cov=n_cov, beam=beam, c_avg=None,
                               return_h = return_h)

    # The beamformer weights matrix w is what we need for waveforms reconstruction;
    # estimated source orientations are used hear just to compare those to the
    # true ones. 
    w, u = res[:2]

    # Get the scalar forward solutionsi for the reconstructed sources, if requested,
    # and verify that those are correct. NOTE that they won't match h_mtx for
    # n_src > 1, because single source orientations won't be accurate
    if return_h:
        h = res[2]
        assert np.allclose(h, np.einsum('scv,vs->cs',fs,u))

    # Verification. Print the true source orientations:
    print("True source orientations:\n", u_src)

    # Print the beamformer-found source orientations:
    print("\nSingle source orientations using true noise covariance\n"
        "- ok not to match the true ones if n_src > 1:\n", u)

    '''
    ---------------------------------------------------------------------------
    In practice the noise covariance n_cov can be hard to estimate from the data
    and of course the true n_cov is only available in simulations.

    Below we construct a diagonal noise covariance as a substitute for the unknown
    n_cov, based on already regularized full covariance r_cov. We'll use half of
    the minimal eigenvalue of the regularized r_cov for diagonal loading.
    ---------------------------------------------------------------------------
    '''
    n_diag = 0.5*np.min(np.linalg.eigvalsh(r_cov))*np.eye(n_chan)

    # Asserting that r_cov - n_cov (or r_cov - n_diag) remains positive definite
    # is not necessary for the weights calculations as such, and does not affect
    # the shapes of reconstructed source waveforms.  However, violating
    # this condition may result in getting negative source powers or source level
    # SNRs.
    assert is_pos_def(r_cov - n_diag), 'The difference (r_cov-n_diag) should be a non-negatively defined matrix'

    '''
    ------------------------------------------------------------------------------
    With the substitute noise covariance ready, calculate the weights and estimate
    source orientations. Now we should not expect an exact match with the true ones
    even when n_src = 1.
    ------------------------------------------------------------------------------
    '''
    res = construct_single_source_weights(fs, r_inv, n_cov=n_diag, beam=beam, c_avg=None,
                               return_h = return_h)

    # Recall that w = res[0]; u = res[1]

    print("\nSingle source orientations using a diagonal noise covariance\n"
        "- ok not to match the true ones even for n_src = 1:\n", res[1])

def construct_test_data(n_src, n_chan, data_rank, snr_db, rng):
    """
    Generate the covariances and forward solutions from the random surrogate
    time courses.

    The data are initially generated in a full-rank subspace of dimension
    ``data_rank`` and then projected to a sensor space of size ``n_chan``.

    Parameters
    ----------
    n_src : int
        Number of sources.
    n_chan : int
        Number of sensor channels.
    data_rank : int
        Effective data rank used to generate the full-rank subspace data.
    snr_db : float
        Target sensor-level power SNR in dB.
    rng : numpy.random.Generator-like
        Random generator instance used for reproducible sampling.

    Returns
    -------
    fs : ndarray, shape (n_src, n_chan, 3)
        Vector forward solutions.
    u_src : ndarray, shape (3, n_src)
        Ground-truth source orientations.
    h_mtx : ndarray, shape (n_chan, n_src)
        Scalar forward solutions from ``fs`` and ``u_src``.
    n_cov : ndarray, shape (n_chan, n_chan)
        Noise covariance matrix.
    r_cov : ndarray, shape (n_chan, n_chan)
        Data covariance matrix scaled to the requested SNR.
    """
    # Construct non-degenerate full rank dataset with dims data_rank x ns
    ns = 3*n_src + 100 * n_chan
    data = rng.standard_normal((data_rank, ns))

    # Use 1st 3*n_src columns as our random "vector" forward solutions
    fs = data[:, :3*n_src]
    noise = data[:, 3*n_src:]  # noise is data_rank x 100*nchan

    # Now we need to reshape fs which is (data_rank, 3*n_src) to the desired format
    # (n_src, data_rank, 3):
    fs = fs.reshape(data_rank, n_src, 3).swapaxes(0, 1)  # Verified!

    # Generate random source orientation vectors
    u_src = rng.standard_normal((3, n_src))

    for i in range(n_src):
        u_src[:, i] = u_src[:, i] / norm(u_src[:, i])

    # Generate matrix h_mtx with source "scalar" FS as columns
    # fs is (n_src x data_rank x 3), u = (3 x n_src), so
    # to get h_mtx = data_rank x n_src we have
    h_mtx = np.einsum('scv,vs->cs', fs, u_src)

    # Construct the noise covariance
    n_cov = noise @ noise.T      # Thus we make n_cov positively defined

    # This is H*H^T - an unscaled source covariance, assuming
    # that all sources are uncorrelated and have amp = 1
    HHT = h_mtx @ h_mtx.T
    r_cov = HHT

    # Scale it to desired sensor level SNR
    current_snr = trace(r_cov) / trace(n_cov)
    snr_pwr = float_power(10, 0.1*snr_db)  # Power SNR in linear units
    factor = snr_pwr / current_snr

    # This is a full data cov with requested SNR in reduced space
    r_cov = factor * r_cov + n_cov

    # r_cov, n_cov are non-degenerate matrices of rank data_rank
    # Now upload everything to a sensor space of n_chan dimensions
    # where n_chan > data_rank using a random orthonormal basis in
    # sensor space
    if data_rank < n_chan:
        U = random_orthonormal_columns(n_chan, data_rank, rng)
        r_cov = U @ r_cov @ U.T
        n_cov = U @ n_cov @ U.T
        h_mtx = U @ h_mtx
        fs = np.einsum('cr,srv->scv', U, fs)

    return fs, u_src, h_mtx, n_cov, r_cov

def random_orthonormal_columns(r, c, rng):
    """
    Return an (r x c) random matrix with orthonormal columns.

    Parameters
    ----------
    r : int
        Number of rows. Must satisfy r > c.
    c : int
        Number of columns.
    rng : numpy.random.Generator-like
        Random generator instance that provides ``standard_normal``.
    """
    if not (isinstance(r, int) and isinstance(c, int)):
        raise TypeError("r and c must be integers")
    if r <= 0 or c <= 0:
        raise ValueError("r and c must be positive")
    if r <= c:
        raise ValueError("r must be strictly greater than c")
    if rng is None or not hasattr(rng, "standard_normal"):
        raise TypeError("rng must provide a standard_normal method")

    a = rng.standard_normal((r, c))
    q, r_mtx = np.linalg.qr(a, mode="reduced")

    # Fix signs to keep outputs deterministic for a fixed RNG sequence.
    signs = np.sign(np.diag(r_mtx))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return q

if __name__ == '__main__': 
    main()

