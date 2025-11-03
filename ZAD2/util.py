import numpy as np
from Float import Float

def random_matrix(n: int, m: int):
    matrix = np.random.random((n, m)) + 0.00000001
    return np.frompyfunc(Float, 1, 1)(matrix)

def rand_invertible(n: int) -> np.ndarray:
    while True:
        A = np.random.random((n, n)) + 1e-8
        try:
            np.linalg.inv(A)
            return A
        except np.linalg.LinAlgError:
            continue