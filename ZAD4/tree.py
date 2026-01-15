import numpy as np
from node import Node
from sklearn.utils.extmath import randomized_svd

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
    U, S, V = randomized_svd(M, rank)
    return Node(U, S, V)

def createTree(M, max_rank, eps):
    compressed = compressMatrix(M, max_rank, eps)
    if min(M.shape) // 2 <= max_rank or compressed.S[-1] <= eps:
        return compressed.normalize(eps)
    
    mid_row, mid_col = M.shape[0] // 2, M.shape[1] // 2
    children = [
        createTree(M[:mid_row, :mid_col], max_rank, eps),
        createTree(M[:mid_row, mid_col:], max_rank, eps),
        createTree(M[mid_row:, :mid_col], max_rank, eps),
        createTree(M[mid_row:, mid_col:], max_rank, eps)
    ]

    return Node(children=children)