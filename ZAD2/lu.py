from Float import Float
import numpy as np
from typing import Tuple
from inverse import *

def lu_factorization(A: np.ndarray, mul) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    if n == 1:
        return np.array([[Float(1)]], dtype=Float), np.array([[A[0, 0]]], dtype=Float)
    
    k = n // 2
    L11, U11 = lu_factorization(A[:k, :k], mul = mul)
    L21 = A[k:, :k] @ inverse(U11, mul = mul, triangular=Triangular.UPPER)
    U12 = mul (inverse(L11, mul = mul, triangular=Triangular.LOWER) , A[:k, k:])
    LS, US = lu_factorization(A[k:, k:] - mul(L21 , U12), mul = mul)

    Ltop = np.hstack((L11, np.vectorize(Float)(np.zeros((k, n - k), dtype=Float))), dtype=Float)
    Lbot = np.hstack((L21, LS), dtype=Float)

    Rtop = np.hstack((U11, U12), dtype=Float)
    Rbot = np.hstack((np.vectorize(Float)(np.zeros((n - k, k), dtype=Float)), US), dtype=Float)

    return np.vstack((Ltop, Lbot), dtype=Float), np.vstack((Rtop, Rbot), dtype=Float)