from Float import Float
import numpy as np
from util import random_matrix
from inverse import inverse
from lu import lu_factorization

A = random_matrix(2, 2)

Float.reset()
print(Float.add_counter, Float.sub_counter, Float.mul_counter, Float.div_counter)
L = inverse(A)
print(Float.add_counter, Float.sub_counter, Float.mul_counter, Float.div_counter)
print(type(L[0,0]))