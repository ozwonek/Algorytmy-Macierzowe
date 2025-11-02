import numpy as np
import pytest
from inverse import inverse

EPS_ATOL = 1e-6
EPS_RTOL = 1e-9

def _rand_invertible(n: int) -> np.ndarray:
    while True:
        A = np.random.random((n, n)) + 1e-8
        try:
            np.linalg.inv(A)
            return A
        except np.linalg.LinAlgError:
            continue

@pytest.mark.parametrize("n", [n for n in range(1, 41)])
def test_inverse_matches_numpy(n: int):
    A = _rand_invertible(n)
    inv_custom = np.array(inverse(A, return_type=np.float64), dtype=float)

    I_approx = A @ inv_custom
    err = np.max(np.abs(I_approx - np.eye(n)))
    assert np.allclose(I_approx, np.eye(n), atol=EPS_ATOL, rtol=EPS_RTOL), f"||A*inv - I||_inf = {err}"