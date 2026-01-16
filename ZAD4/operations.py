import numpy as np
from sklearn.utils.extmath import randomized_svd
from node import Node

def matrix_vector_mult(node, vector):
    if len(node.children) == 0:
        return node.U @ (np.diag(node.S) @ (node.V @ vector))

    mid = vector.shape[0] // 2
    top = matrix_vector_mult(node.children[0], vector[:mid]) + matrix_vector_mult(node.children[1], vector[mid:])
    bot = matrix_vector_mult(node.children[2], vector[:mid]) + matrix_vector_mult(node.children[3], vector[mid:])
    return np.vstack((top, bot))

def compressMatrix(M, rank):
    U, S, V = randomized_svd(M, rank)
    return Node(U, S, V)

import numpy as np

def _resize(node):
    node.U = np.asarray(node.U)
    node.V = np.asarray(node.V)
    node.S = np.asarray(node.S).reshape(-1)

    if node.U.ndim == 1:
        node.U = node.U.reshape(-1, 1)     
    elif node.U.ndim == 0:
        node.U = node.U.reshape(1, 1)

    if node.V.ndim == 1:
        node.V = node.V.reshape(1, -1)    
    elif node.V.ndim == 0:
        node.V = node.V.reshape(1, 1)

    r = min(node.U.shape[1], node.S.shape[0], node.V.shape[0])
    if r < 1:
        n = node.U.shape[0]
        m = node.V.shape[1]
        node.U = np.zeros((n, 1))
        node.S = np.zeros((1,))
        node.V = np.zeros((1, m))
        return

    node.U = node.U[:, :r]
    node.S = node.S[:r]
    node.V = node.V[:r, :]

def _match_rows(A, B):
    n = max(A.shape[0], B.shape[0])
    if A.shape[0] != n:
        A = np.resize(A, (n, A.shape[1]))
    if B.shape[0] != n:
        B = np.resize(B, (n, B.shape[1]))
    return A, B

def _match_cols(A, B):
    m = max(A.shape[1], B.shape[1])
    if A.shape[1] != m:
        A = np.resize(A, (A.shape[0], m))
    if B.shape[1] != m:
        B = np.resize(B, (B.shape[0], m))
    return A, B


def matrix_matrix_add(node1, node2):
    if len(node1.children) == 0 and len(node2.children) == 0:
        _resize(node1)
        _resize(node2)

        A = node1.U @ np.diag(node1.S)
        B = node2.U @ np.diag(node2.S)

        A, B = _match_rows(A, B)          
        V1, V2 = _match_cols(node1.V, node2.V)

        U = np.hstack((A, B))
        V = np.vstack((V1, V2))
        r = max(node1.S.shape[0], node2.S.shape[0])
        return compressMatrix(U @ V, r)
    elif len(node1.children) == 0 or len(node2.children) == 0:
        leaf = node1 if len(node1.children) == 0 else node2
        other = node2 if len(node1.children) == 0 else node1

        n, r, m = leaf.U.shape[0], leaf.S.shape[0], leaf.V.shape[1]
        U = leaf.U @ np.diag(leaf.S)
        U1 = U[:n // 2]
        U2 = U[n // 2:]
        V1 = leaf.V[:, :m // 2]
        V2 = leaf.V[:, m // 2:]

        root = Node()
        root.children.append(matrix_matrix_add(Node(U1, np.ones(r), V1), other.children[0]))
        root.children.append(matrix_matrix_add(Node(U1, np.ones(r), V2), other.children[1]))
        root.children.append(matrix_matrix_add(Node(U2, np.ones(r), V1), other.children[2]))
        root.children.append(matrix_matrix_add(Node(U2, np.ones(r), V2), other.children[3]))
        return root
    else:
        root = Node()
        root.children.append(matrix_matrix_add(node1.children[0], node2.children[0]))
        root.children.append(matrix_matrix_add(node1.children[1], node2.children[1]))
        root.children.append(matrix_matrix_add(node1.children[2], node2.children[2]))
        root.children.append(matrix_matrix_add(node1.children[3], node2.children[3]))
        return root

def matrix_matrix_mul(node1, node2):
    if len(node1.children) == 0 and len(node2.children) == 0:
        if (len(node1.S) == 1 and node1.S[0] == 0) or (len(node2.S) == 1 and node2.S[0] == 0):
            return Node(np.zeros((1, 1)), np.zeros((1)), np.zeros((1, 1)))

        return Node(node1.U @ (np.diag(node1.S) @ (node1.V @ node2.U)), node2.S, node2.V)
    elif len(node1.children) == 0:
        n, r, m = node1.U.shape[0], node1.S.shape[0], node1.V.shape[1]
        U = node1.U @ np.diag(node1.S)
        U1 = U[:n // 2]
        U2 = U[n // 2:]
        V1 = node1.V[:, :m // 2]
        V2 = node1.V[:, m // 2:]

        root = Node()
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(Node(U1, np.ones(r), V1), node2.children[0]),
            matrix_matrix_mul(Node(U1, np.ones(r), V2), node2.children[2])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(Node(U1, np.ones(r), V1), node2.children[1]),
            matrix_matrix_mul(Node(U1, np.ones(r), V2), node2.children[3])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(Node(U2, np.ones(r), V1), node2.children[0]),
            matrix_matrix_mul(Node(U2, np.ones(r), V2), node2.children[2])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(Node(U2, np.ones(r), V1), node2.children[1]),
            matrix_matrix_mul(Node(U2, np.ones(r), V2), node2.children[3])
        ))
        return root
    elif len(node2.children) == 0:
        n, r, m = node2.U.shape[0], node2.S.shape[0], node2.V.shape[1]
        U = node2.U @ np.diag(node2.S)
        U1 = U[:n // 2]
        U2 = U[n // 2:]
        V1 = node2.V[:, :m // 2]
        V2 = node2.V[:, m // 2:]

        root = Node()
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[0], Node(U1, np.ones(r), V1)),
            matrix_matrix_mul(node1.children[1], Node(U2, np.ones(r), V1))
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[0], Node(U1, np.ones(r), V2)),
            matrix_matrix_mul(node1.children[1], Node(U2, np.ones(r), V2))
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[2], Node(U1, np.ones(r), V1)),
            matrix_matrix_mul(node1.children[3], Node(U2, np.ones(r), V1))
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[2], Node(U1, np.ones(r), V2)),
            matrix_matrix_mul(node1.children[3], Node(U2, np.ones(r), V2))
        ))
        return root
    else:
        root = Node()
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[0], node2.children[0]),
            matrix_matrix_mul(node1.children[1], node2.children[2])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[0], node2.children[0]),
            matrix_matrix_mul(node1.children[1], node2.children[3])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[2], node2.children[0]),
            matrix_matrix_mul(node1.children[3], node2.children[2])
        ))
        root.children.append(matrix_matrix_add(
            matrix_matrix_mul(node1.children[2], node2.children[1]),
            matrix_matrix_mul(node1.children[3], node2.children[3])
        ))
        return root