import numpy as np
from Float import Float

def random_matrix(n: int):
    matrix = np.random.random((n, n)) + 0.00000001
    return np.frompyfunc(Float, 1, 1)(matrix)

def random_lower_triangular(n: int):
    mat = np.random.random((n, n)) + 0.00000001
    lower = np.tril(mat)
    return np.frompyfunc(Float, 1, 1)(lower)

def random_upper_triangular(n: int):
    mat = np.random.random((n, n)) + 0.00000001
    upper = np.triu(mat)
    return np.frompyfunc(Float, 1, 1)(upper)

def random_vector_T(n: int):
    matrix = np.random.random((n, 1)) + 0.00000001
    return np.frompyfunc(Float, 1, 1)(matrix)