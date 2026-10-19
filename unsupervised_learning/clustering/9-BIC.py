#!/usr/bin/env python3
"""Determines the best number of clusters using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not np.all(np.isfinite(X))
            or isinstance(kmin, bool)
            or not isinstance(kmin, int) or kmin <= 0
            or (kmax is not None and
                (isinstance(kmax, bool)
                 or not isinstance(kmax, int) or kmax <= 0))
            or isinstance(iterations, bool)
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float))
            or isinstance(tol, bool) or not np.isfinite(tol) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmax < kmin or kmax > n:
        return None, None, None, None

    log_likelihoods = np.zeros(kmax - kmin + 1)
    bics = np.zeros(kmax - kmin + 1)

    best_k = None
    best_result = None
    best_bic = np.inf

    for idx, k in enumerate(range(kmin, kmax + 1)):
        try:
            result = expectation_maximization(
                X, k, iterations, tol, verbose
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None, None, None, None

        if (result is None or len(result) != 5
                or any(value is None for value in result)):
            return None, None, None, None

        pi, m, S, g, log_likelihood = result

        if (not np.isfinite(log_likelihood)
                or not np.all(np.isfinite(pi))
                or not np.all(np.isfinite(m))
                or not np.all(np.isfinite(S))
                or not np.all(np.isfinite(g))):
            return None, None, None, None

        log_likelihoods[idx] = log_likelihood

        p = (k - 1) + k * d + k * d * (d + 1) // 2
        bics[idx] = p * np.log(n) - 2 * log_likelihood

        if bics[idx] < best_bic:
            best_bic = bics[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
