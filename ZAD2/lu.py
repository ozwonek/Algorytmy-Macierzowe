from Float import Float
import numpy as np
from typing import Tuple
from inverse import inverse

def lu_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    if n == 1:
        return np.array([[Float(1)]], dtype=Float), np.array([[A[0, 0]]], dtype=Float)
    
    k = n // 2
    L11, U11 = lu_factorization(A[:k, :k])
    U11_inv = inverse(U11)
    L11_inv = inverse(L11)
    S = A[k:, k:] - A[k:, :k] @ U11_inv @ L11_inv @ A[:k, k:]
    LS, US = lu_factorization(S)

    L21 = A[k:, :k] @ U11_inv
    U12 = L11_inv @ A[:k, k:]

    Ltop = np.hstack((L11, np.vectorize(Float)(np.zeros((k, n - k), dtype=Float))), dtype=Float)
    Lbot = np.hstack((L21, LS), dtype=Float)

    Rtop = np.hstack((U11, U12), dtype=Float)
    Rbot = np.hstack((np.vectorize(Float)(np.zeros((n - k, k), dtype=Float)), US), dtype=Float)

    return np.vstack((Ltop, Lbot), dtype=Float), np.vstack((Rtop, Rbot), dtype=Float)