import numpy as np
import matplotlib.pyplot as plt
from tree import *

def visualizeCompression(node, M, row1, row2, col1, col2):
    if len(node.children) == 0:
        rank = len(node.S)
        M[row1 : row2, col1 : col1 + rank] = 0
        M[row1 : row1 + rank, col1 : col2] = 0
        return
    
    row_mid, col_mid = (row1 + row2) // 2, (col1 + col2) // 2
    visualizeCompression(node.children[0], M, row1, row_mid, col1, col_mid)
    visualizeCompression(node.children[1], M, row1, row_mid, col_mid, col2)
    visualizeCompression(node.children[2], M, row_mid, row2, col1, col_mid)
    visualizeCompression(node.children[3], M, row_mid, row2, col_mid, col2)
