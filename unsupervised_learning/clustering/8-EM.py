#!/usr/bin/env python3
"""Expectation maximization algorithm with GMM."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Performs expectation maximization for a GMM."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(k, int) or k <= 0
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(
            round(log_likelihood, 5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, new_likelihood = expectation(X, pi, m, S)
        if g is None or new_likelihood is None:
            return None, None, None, None, None

        converged = abs(new_likelihood - log_likelihood) <= tol

        if verbose and (i % 10 == 0 or i == iterations or converged):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(new_likelihood, 5)))

        log_likelihood = new_likelihood
        if converged:
            break

    return pi, m, S, g, log_likelihood
