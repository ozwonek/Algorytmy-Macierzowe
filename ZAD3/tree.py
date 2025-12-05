import numpy as np
from node import Node

def reconstructMatrix(node):
    if len(node.children) == 0:
        return node.U @ np.diag(node.S) @ node.V

    top_left = reconstructMatrix(node.children[0])
    top_right = reconstructMatrix(node.children[1])
    bot_left = reconstructMatrix(node.children[2])
    bot_right = reconstructMatrix(node.children[3])

    top = np.hstack((top_left, top_right))
    bot = np.hstack((bot_left, bot_right))
    return np.vstack((top, bot))

def compressMatrix(M, rank, eps):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    S = S[S > eps]
    rank = min(rank, S.shape[0])
    
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]

    return Node(U, S, V) 

def computeError(M_original, node_compressed):
    M_compressed = reconstructMatrix(node_compressed)
    num = np.linalg.norm(M_original - M_compressed, ord='fro')
    den = np.linalg.norm(M_original, ord='fro')
    if den < 1e-12:
        return 0.0 if num < 1e-12 else 1.0
    return num / den

def createTree(M, max_rank, eps):
    compressed = compressMatrix(M, max_rank, eps)
    if min(M.shape) <= max_rank or computeError(M, compressed) <= eps:
        return compressed
    
    mid_row, mid_col = M.shape[0] // 2, M.shape[1] // 2
    children = [
        createTree(M[:mid_row, :mid_col], max_rank, eps),
        createTree(M[:mid_row, mid_col:], max_rank, eps),
        createTree(M[mid_row:, :mid_col], max_rank, eps),
        createTree(M[mid_row:, mid_col:], max_rank, eps)
    ]

    return Node(children=children)