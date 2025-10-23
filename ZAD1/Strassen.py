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

        P1 = strassen_with_padding(X11 + X22, Y11 + Y22) 
        P2 = strassen_with_padding(X21 + X22, Y11)
        P3 = strassen_with_padding(X11, Y12 - Y22)
        P4 = strassen_with_padding(X22, Y21 - Y11)
        P5 = strassen_with_padding(X11 + X12, Y22)
        P6 = strassen_with_padding(X21 - X11, Y11 + Y12)
        P7 = strassen_with_padding(X12 - X22, Y21 + Y22)

        Z11 = P1 + P4 - P5 + P7
        Z12 = P3 + P5
        Z21 = P2 + P4
        Z22 = P1 - P2 + P3 + P6

        top = np.hstack((Z11, Z12), dtype=Float)
        bot = np.hstack((Z21, Z22), dtype=Float)
        return np.vstack((top, bot), dtype=Float)
    
    return _mul(A_pad, B_pad)[:n, :m]

def strassen_without_padding(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    if n == 1:
        return A @ B
    
    if n % 2 != 0:
        m = n - 1

        A11 = A[:m, :m]
        a12 = A[:m, m:]
        a21 = A[m:, :m]
        a22 = A[m:, m:]

        B11 = B[:m, :m]
        b12 = B[:m, m:]
        b21 = B[m:, :m]
        b22 = B[m:, m:]

        C11 = strassen_without_padding(A11, B11) + a12 @ b21
        C12 = A11 @ b12 + a12 @ b22
        C21 = a21 @ B11 + a22 @ b21
        C22 = a21 @ b12 + a22 @ b22

        top = np.hstack((C11, C12), dtype=Float)
        bot = np.hstack((C21, C22), dtype=Float)
        return np.vstack((top, bot), dtype=Float)
    
    n //= 2
    X11 = A[:n, :n]
    X12 = A[:n, n:]
    X21 = A[n:, :n] 
    X22 = A[n:, n:]
    Y11 = B[:n, :n] 
    Y12 = B[:n, n:]
    Y21 = B[n:, :n] 
    Y22 = B[n:, n:]

    P1 = strassen_without_padding(X11 + X22, Y11 + Y22) 
    P2 = strassen_without_padding(X21 + X22, Y11)
    P3 = strassen_without_padding(X11, Y12 - Y22)
    P4 = strassen_without_padding(X22, Y21 - Y11)
    P5 = strassen_without_padding(X11 + X12, Y22)
    P6 = strassen_without_padding(X21 - X11, Y11 + Y12)
    P7 = strassen_without_padding(X12 - X22, Y21 + Y22)

    Z11 = P1 + P4 - P5 + P7
    Z12 = P3 + P5
    Z21 = P2 + P4
    Z22 = P1 - P2 + P3 + P6

    top = np.hstack((Z11, Z12), dtype=Float)
    bot = np.hstack((Z21, Z22), dtype=Float)
    return np.vstack((top, bot), dtype=Float)
        
    
