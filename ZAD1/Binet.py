import numpy as np
from Float import Float
from benchmark import benchmark
from helpers import multiply_matrices, add_padding

@benchmark(reference_func=multiply_matrices)
def binet_with_padding_benchmark(A: np.ndarray, B: np.ndarray):
    return binet_with_padding(A, B)

@benchmark(reference_func=multiply_matrices)
def binet_without_padding_benchmark(A: np.ndarray, B: np.ndarray):
    return binet_without_padding(A, B)

@benchmark(reference_func=multiply_matrices)
def binet_two_split_benchmark(A: np.ndarray, B: np.ndarray):
    return binet_two_split(A, B)

def binet_with_padding(A: np.ndarray, B: np.ndarray):
    A_pad, B_pad, n, m = add_padding(A, B)

    def _mul(X: np.ndarray, Y: np.ndarray):
        n = X.shape[0]
        if n == 1:
            return X @ Y

        n //= 2

        X11 = X[:n, :n]
        X12 = X[:n, n:]
        X21 = X[n:, :n] 
        X22 = X[n:, n:]
        Y11 = Y[:n, :n] 
        Y12 = Y[:n, n:]
        Y21 = Y[n:, :n] 
        Y22 = Y[n:, n:]

        Z11 = _mul(X11, Y11) + _mul(X12, Y21)
        Z12 = _mul(X11, Y12) + _mul(X12, Y22)
        Z21 = _mul(X21, Y11) + _mul(X22, Y21)
        Z22 = _mul(X21, Y12) + _mul(X22, Y22)

        top = np.hstack((Z11, Z12), dtype=Float)
        bot = np.hstack((Z21, Z22), dtype=Float)
        return np.vstack((top, bot), dtype=Float)
    
    return _mul(A_pad, B_pad)[:n, :m]


def binet_without_padding(X: np.ndarray, Y: np.ndarray):
    n, k = X.shape
    _, m = Y.shape

    if min(n, k, m) == 1:
        return X @ Y
    
    n //= 2
    m //= 2
    k //= 2

    X11 = X[:n, :k]
    X12 = X[:n, k:]
    X21 = X[n:, :k]
    X22 = X[n:, k:]
    Y11 = Y[:k, :m]
    Y12 = Y[:k, m:]
    Y21 = Y[k:, :m]
    Y22 = Y[k:, m:]

    Z11 = binet_without_padding(X11, Y11) + binet_without_padding(X12, Y21)
    Z12 = binet_without_padding(X11, Y12) + binet_without_padding(X12, Y22)
    Z21 = binet_without_padding(X21, Y11) + binet_without_padding(X22, Y21)
    Z22 = binet_without_padding(X21, Y12) + binet_without_padding(X22, Y22)

    top = np.hstack((Z11, Z12), dtype=Float)
    bot = np.hstack((Z21, Z22), dtype=Float)
    return np.vstack((top, bot), dtype=Float)        


def binet_two_split(X: np.ndarray, Y: np.ndarray):
    n, k = X.shape
    _, m = Y.shape

    if min(n, k, m) == 1:
        return X @ Y
    
    mx = max(n, k, m)
    if mx == n:
        n //= 2
        return np.vstack(((binet_two_split(X[:n, :], Y), binet_two_split(X[n:, :], Y))), dtype=Float)
    elif mx == m:
        m //= 2
        return np.hstack(((binet_two_split(X, Y[:, :m]), binet_two_split(X, Y[:, m:]))), dtype=Float)
    else:
        k //= 2
        return binet_two_split(X[:, :k], Y[:k, :]) + binet_two_split(X[:, k:], Y[k:, :])