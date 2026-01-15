import numpy as np

def matrix_mul_vector(node, vector):
    if len(node.children) == 0:
        return node.U @ (np.diag(node.S) @ (node.V @ vector))

    mid = vector.shape[0] // 2
    top = matrix_mul_vector(node.children[0], vector[:mid]) + matrix_mul_vector(node.children[1], vector[mid:])
    bot = matrix_mul_vector(node.children[2], vector[:mid]) + matrix_mul_vector(node.children[3], vector[mid:])
    return np.vstack((top, bot))