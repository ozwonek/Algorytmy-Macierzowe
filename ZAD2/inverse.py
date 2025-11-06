import numpy as np
from Float import Float
from enum import Enum

class Triangular(Enum):
    NONE = 1
    UPPER = 2
    LOWER = 3

def inverse(A: np.ndarray, triangular: Triangular = Triangular.NONE) -> np.ndarray:
    n = A.shape[0]

    if n == 1:
        return np.array([[1.0 / A[0, 0]]], dtype=Float)
    
    k = n // 2
    A11_inv = inverse(A[:k, :k])

    match triangular:
        case Triangular.LOWER | Triangular.UPPER:
            S22_inv = inverse(A[k:, k:])
            C11 = A11_inv
            if triangular == Triangular.LOWER:
                C12 = np.vectorize(Float)(np.zeros((k, n - k), dtype=Float))
                C21 = -S22_inv @ A[k:, :k] @ A11_inv
            elif triangular == Triangular.UPPER:
                C12 =  -A11_inv @ A[:k, k:] @  S22_inv
                C21 = np.vectorize(Float)(np.zeros((n - k, k), dtype=Float))
        case Triangular.NONE:
            S22_inv = inverse(A[k:, k:] -A[k:, :k] @ A11_inv @ A[:k, k:])
            X = A11_inv @ A[:k, k:] @ S22_inv
            Y = A[k:, :k] @ A11_inv
            C11 = A11_inv + X @ Y
            C12 = -X
            C21 = -S22_inv @ Y
    
    top = np.hstack((C11, C12), dtype=Float)
    bot = np.hstack((C21, S22_inv), dtype=Float)
    return np.vstack((top, bot), dtype=Float)