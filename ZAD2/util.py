import numpy as np
from Float import Float

def random_matrix(n: int, m: int, return_type=Float):
    matrix = np.random.random((n, m)) + 0.00000001
    return np.frompyfunc(return_type, 1, 1)(matrix)
