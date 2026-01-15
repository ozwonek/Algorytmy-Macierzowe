import numpy as np
from tree import *
from utils import *
from operations import *
from plots import *

# M = generate_3d_crate_matrix(4)
# root = createTree(M, 16, 0.6)

# T = np.ones(M.shape)
# visualizeCompression(root, T, 0, T.shape[0], 0, T.shape[1])

# plt.imshow(T, cmap='grey')
# plt.show()

# vector = np.random.rand(M.shape[0], 1)

# print(M.shape)
# print(calculate_error(M @ vector, matrix_vector_mult(root, vector)))

# print(np.ones(4))

M1 = generate_3d_crate_matrix(2)
# M2 = generate_3d_crate_matrix(4)

# print(M1 + M2)
# print(M1.shape)
root1 = createTree(M1, max(4, int(M1.shape[0]**0.5)), 0.7)
# root2 = createTree(M1, M1.shape[0] // 4, 0.8)

# print(reconstructMatrix(root1))
# print(reconstructMatrix(root2))
# print(M1 @ M1)
# print(reconstructMatrix(matrix_matrix_mul(root1, root2)))

from time import perf_counter
t0 = perf_counter()
print(calculate_error(M1 @ M1, reconstructMatrix(matrix_matrix_mul(root1, root1))))
t1 = perf_counter()
print(t1 - t0)
