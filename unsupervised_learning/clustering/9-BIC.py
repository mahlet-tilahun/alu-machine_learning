#!/usr/bin/env python3
"""Determines the best number of clusters using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X (numpy.ndarray): Dataset with shape (n, d).
        kmin (int): Minimum number of clusters to test.
        kmax (int or None): Maximum number of clusters to test.
        iterations (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance.
        verbose (bool): Whether to print EM progress.

    Returns:
        tuple: Best k, best GMM parameters, log likelihoods, and BICs.
            Returns four None values if the inputs or EM results fail.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0):
        return None, None, None, None

    if (not isinstance(kmin, int) or isinstance(kmin, bool)
            or kmin <= 0):
        return None, None, None, None

    if kmax is not None:
        if (not isinstance(kmax, int) or isinstance(kmax, bool)
                or kmax <= 0 or kmax < kmin):
            return None, None, None, None

    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations <= 0):
        return None, None, None, None

    if (not isinstance(tol, (int, float)) or isinstance(tol, bool)
            or tol < 0 or not np.isfinite(tol)):
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin > n or kmax < kmin or kmax > n:
        return None, None, None, None

    log_likelihoods = np.zeros(kmax - kmin + 1)
    bics = np.zeros(kmax - kmin + 1)

    best_k = None
    best_result = None
    best_bic = np.inf

    for idx, k in enumerate(range(kmin, kmax + 1)):
        result = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )

        if result is None or len(result) != 5:
            return None, None, None, None

        pi, m, S, g, log_likelihood = result

        if (pi is None or m is None or S is None
                or g is None or log_likelihood is None):
            return None, None, None, None

        if not np.isfinite(log_likelihood):
            return None, None, None, None

        log_likelihoods[idx] = log_likelihood

        p = (k - 1) + k * d + k * d * (d + 1) // 2
        bics[idx] = p * np.log(n) - 2 * log_likelihood

        if bics[idx] < best_bic:
            best_bic = bics[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
