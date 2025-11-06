import numpy as np
import pytest
from inverse import *
from lu import *
from util import *
from gauss import *


EPS_ATOL = 1e-5
EPS_RTOL = 1e-9

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_inverse(n: int):
    A = random_matrix(n)
    inv_custom = inverse(A,)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_inverse_lower_triangular(n: int):
    A = random_lower_triangular(n)
    inv_custom = inverse(A, triangular=Triangular.LOWER)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_inverse_upper_triangular(n: int):
    A = random_upper_triangular(n)
    inv_custom = inverse(A, triangular=Triangular.UPPER )

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_lu_factorization(n :int):
    A = random_matrix(n)
    L, U = lu_factorization(A, )

    A_approx = L @ U
    err = np.max(np.abs(A_approx - A))
    assert np.allclose(A_approx.astype(np.float64), A.astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(L*U - A) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_gauss_check_if_upper(n: int):
    A = random_matrix(n)
    b = random_vector_T(n)

    C, _ = gauss_elimination(A, b,)
    
    for i in range(n):
        for j in range(i):
            assert abs(float(C[i, j])) < EPS_ATOL, f"Element C[{i},{j}] = {C[i,j]} should be zero (upper triangular matrix)"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_gauss(n: int):
    A = random_matrix(n)
    b = random_vector_T(n)
    
    A_original = A.copy()
    b_original = b.copy()
    
    C, b_modified = gauss_elimination(A, b,)
    x_custom = back_substitution(C, b_modified)
    
    A_numpy = A_original.astype(np.float64)
    b_numpy = b_original.astype(np.float64).flatten()
    x_numpy = np.linalg.solve(A_numpy, b_numpy).reshape(-1, 1)
    
    x_custom_float = x_custom.astype(np.float64)
    
    err = np.max(np.abs(x_custom_float - x_numpy))
    assert np.allclose(x_custom_float, x_numpy, atol=EPS_ATOL, rtol=EPS_RTOL), f"max(x_custom - x_numpy) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 21)])
def test_determinant(n: int):
    A = random_matrix(n)    
    det_custom = determinant(A)    
    A_numpy = A.astype(np.float64)
    det_numpy = np.linalg.det(A_numpy)    
    det_custom_float = float(det_custom)    
    err = abs(det_custom_float - det_numpy)
    assert np.allclose(det_custom_float, det_numpy, atol=EPS_ATOL, rtol=EPS_RTOL), f"Custom det = {det_custom_float}, NumPy det = {det_numpy}, error = {err}"