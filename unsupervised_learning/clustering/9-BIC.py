#!/usr/bin/env python3
"""Determines the best number of clusters using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(kmin, int) or kmin <= 0
            or (kmax is not None and
                (not isinstance(kmax, int) or kmax <= 0))
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin > kmax or kmax > n:
        return None, None, None, None

    log_likelihoods = np.zeros(kmax - kmin + 1)
    bics = np.zeros(kmax - kmin + 1)

    best_k = None
    best_result = None
    best_bic = np.inf

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if (pi is None or m is None or S is None or
                g is None or log_likelihood is None):
            return None, None, None, None

        log_likelihoods[idx] = log_likelihood

        # Number of parameters: priors + means + covariance matrices.
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        bics[idx] = p * np.log(n) - 2 * log_likelihood

        if bics[idx] < best_bic:
            best_bic = bics[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
