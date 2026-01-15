import numpy as np

class Node:
    def __init__(self, U = None, S = None, V = None, children = None):
        self.U = U
        self.S = S
        self.V = V
        self.children = children if children is not None else []
    
    def normalize(self, eps):
        self.S = self.S[self.S > eps]
        r = self.S.shape[0]
        if r == 0:
            self.S = np.array([0])
            r = 1
        self.U = self.U[:, :r]
        self.V = self.V[:r, :]
        return self