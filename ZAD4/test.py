import numpy as np
from tree import *
from utils import *
from operations import *
from plots import *

M = generate_3d_crate_matrix(3)
root = createTree(M, 2, 32)

# T = np.ones(M.shape)
# visualizeCompression(root, T, 0, T.shape[0], 0, T.shape[1])

# plt.imshow(T, cmap='grey')
# plt.show()

vector = np.random.rand(M.shape[0], 1)

print(M.shape)
print(calculate_error(M @ vector, matrix_vector_mult(root, vector)))