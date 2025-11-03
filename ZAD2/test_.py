import numpy as np
import pytest
from inverse import *
from lu import *
from util import *

EPS_ATOL = 1e-5
EPS_RTOL = 1e-9

@pytest.mark.parametrize("n", [n for n in range(1, 31)])
def test_inverse(n: int):
    A = random_matrix(n)
    inv_custom = inverse(A)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 31)])
def test_inverse_lower_triangular(n: int):
    A = random_lower_triangular(n)
    inv_custom = inverse(A, triangular=Triangular.LOWER)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 31)])
def test_inverse_upper_triangular(n: int):
    A = random_upper_triangular(n)
    inv_custom = inverse(A, triangular=Triangular.UPPER)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx.astype(np.float64), np.eye(n).astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(A*A_inv - I) = {err}"

@pytest.mark.parametrize("n", [n for n in range(1, 31)])
def test_lu_factorization(n :int):
    A = random_matrix(n)
    L, U = lu_factorization(A)

    A_approx = L @ U
    err = np.max(np.abs(A_approx - A))
    assert np.allclose(A_approx.astype(np.float64), A.astype(np.float64), atol=EPS_ATOL, rtol=EPS_RTOL), f"max(L*U - A) = {err}"