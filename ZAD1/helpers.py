import numpy as np
from Float import Float
from benchmark import benchmark


def add_padding(A: np.ndarray, B: np.ndarray):

    n, kA = A.shape
    kB, m = B.shape

    mx = max(n, kA, m)
    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()
    
    next_power_of_two = next_power_of_two(mx)

    A_pad, B_pad = np.zeros((next_power_of_two, next_power_of_two)), np.zeros((next_power_of_two, next_power_of_two))
    A_pad[:n, :kA] = A
    B_pad[:kB, :m] = B
    A_pad = np.frompyfunc(Float, 1, 1)(A_pad)
    B_pad = np.frompyfunc(Float, 1, 1)(B_pad)

    return A_pad, B_pad, n, m 

def multiply_matrices(A, B):
    return A @ B


@benchmark(reference_func=multiply_matrices)
def numpy_benchmark(A, B):
    return multiply_matrices(A, B)
