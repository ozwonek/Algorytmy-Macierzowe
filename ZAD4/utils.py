import numpy as np

def generate_3d_crate_matrix(k):
    N = 2**(k*3)
    M = np.zeros(shape=(N,N))
    p = 2**k
    for x in range(p):
        for y in range(p):
            for z in range(p):
                point = x * p**2 + y * p**1 + z * p**0
                if 0 <= point - p**2 < N: M[point][point - p**2] = np.random.random()
                if 0 <= point + p**2 < N: M[point][point + p**2] = np.random.random()
                if 0 <= point - p**1 < N: M[point][point - p**1] = np.random.random()
                if 0 <= point + p**1 < N: M[point][point + p**1] = np.random.random()
                if 0 <= point - p**0 < N: M[point][point - p**0] = np.random.random()
                if 0 <= point + p**0 < N: M[point][point + p**0] = np.random.random()
    return M