import numpy as np
from Float import Float
from helpers import add_padding

def strassen_with_padding(A: np.ndarray, B: np.ndarray):
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

        P1 = _mul(X11 + X22, Y11 + Y22) 
        P2 = _mul(X21 + X22, Y11)
        P3 = _mul(X11, Y12 - Y22)
        P4 = _mul(X22, Y21 - Y11)
        P5 = _mul(X11 + X12, Y22)
        P6 = _mul(X21 - X11, Y11 + Y12)
        P7 = _mul(X12 - X22, Y21 + Y22)

        Z11 = P1 + P4 - P5 + P7
        Z12 = P3 + P5
        Z21 = P2 + P4
        Z22 = P1 - P2 + P3 + P6

        top = np.hstack((Z11, Z12), dtype=Float)
        bot = np.hstack((Z21, Z22), dtype=Float)
        return np.vstack((top, bot), dtype=Float)
    
    return _mul(A_pad, B_pad)[:n, :m]