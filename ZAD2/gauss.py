import numpy as np
from Float import Float
from inverse import *
from lu import lu_factorization
from typing import Tuple

def gauss_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    if n == 1:
        return np.array([[Float(1)]], dtype=Float), np.array([[b[0, 0] / A[0, 0]]], dtype=Float)
    
    k = n // 2
    L11, U11 = lu_factorization(A[:k, :k])
    L11_inv = inverse(L11, triangular=Triangular.LOWER)
    A11_inv = inverse(A[:k, :k])

    C11 = U11
    C12 = L11_inv @ A[:k, k:]
    C21 = np.vectorize(Float)(np.zeros((n - k, k), dtype=Float))
    S = A[k:, k:] - A[k:, :k] @ A11_inv @ A[:k, k:]
    LS, LU = lu_factorization(S)
    LS_inv = inverse(LS, triangular=Triangular.LOWER)
    C22 = LU

    b1 = L11_inv @ b[:k, :]
    b2 = LS_inv @ (b[k:, :] - A[k:, :k] @ A11_inv @ b[:k, :])

    Ctop = np.hstack((C11, C12), dtype=Float)
    Cbot = np.hstack((C21, C22), dtype=Float)

    C = np.vstack((Ctop, Cbot), dtype=Float)
    b_ans = np.vstack((b1, b2), dtype=Float)
    return C, b_ans

def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    x = np.zeros((n, 1), dtype=Float)
    
    for i in range(n - 1, -1, -1):
        sum_ax = Float(0)
        for j in range(i + 1, n):
            sum_ax += A[i, j] * x[j, 0]        
        x[i, 0] = (b[i, 0] - sum_ax) / A[i, i]
    return x