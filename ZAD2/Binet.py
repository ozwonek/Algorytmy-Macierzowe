import numpy as np
from Float import Float


def binet(X: np.ndarray, Y: np.ndarray):
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

    Z11 = binet(X11, Y11) + binet(X12, Y21)
    Z12 = binet(X11, Y12) + binet(X12, Y22)
    Z21 = binet(X21, Y11) + binet(X22, Y21)
    Z22 = binet(X21, Y12) + binet(X22, Y22)

    top = np.hstack((Z11, Z12), dtype=Float)
    bot = np.hstack((Z21, Z22), dtype=Float)
    return np.vstack((top, bot), dtype=Float)        

