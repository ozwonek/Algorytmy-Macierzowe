import numpy as np
from Float import Float
from inverse import inverse
from lu import lu_factorization

def gauss_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]

    if n == 1:
        return A
    
    k = n // 2
    