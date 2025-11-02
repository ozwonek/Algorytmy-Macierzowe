import numpy as np
from util import *

def inverse(A: np.ndarray, return_type=Float) -> np.ndarray:
    n = A.shape[0]

    if n == 1:
        return np.array([[1.0 / A[0, 0]]])
    
    n //= 2
    A11_inv = inverse(A[:n, :n], return_type=return_type)
    S22_inv = inverse(A[n:, n:] - A[n:, :n] @ A11_inv @ A[:n, n:], return_type=return_type)

    C11 = A11_inv + A11_inv @ A[:n, n:] @ S22_inv @ A[n:, :n] @ A11_inv
    C12 = -A11_inv @ A[:n, n:] @ S22_inv
    C21 = -S22_inv @ A[n:, :n] @ A11_inv
    C22 = S22_inv

    top = np.hstack((C11, C12), dtype=return_type)
    bot = np.hstack((C21, C22), dtype=return_type)
    return np.vstack((top, bot), dtype=return_type)