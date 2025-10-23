import numpy as np
from Float import Float

def binet_with_padding(A: np.ndarray, B: np.ndarray):
    n, kA = A.shape
    kB, m = B.shape

    mx = max(n, kA, m)
    next_pow2 = 1
    while next_pow2 < mx:
        next_pow2 *= 2
    
    A_pad, B_pad = np.zeros((next_pow2, next_pow2)), np.zeros((next_pow2, next_pow2))
    A_pad[:n, :kA] = A
    B_pad[:kB, :m] = B
    A_pad = np.frompyfunc(Float, 1, 1)(A_pad)
    B_pad = np.frompyfunc(Float, 1, 1)(B_pad)

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
    n, kX = X.shape
    kY, m = Y.shape

    if min(n, kX, m) == 1:
        return X @ Y
    
    n //= 2
    m //= 2
    k = kX // 2

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