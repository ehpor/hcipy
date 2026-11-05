import numpy as np
import pytest

from hcipy._math.backends import array_namespace, to_numpy
from hcipy._math.linalg import matmul


@pytest.mark.parametrize('adjoint_a,adjoint_b', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('batch_shape', [(), (2, 3)])
def test_matmul(xp, adjoint_a, adjoint_b, dtype, batch_shape):
    rng = np.random.default_rng(21)
    a_shape = batch_shape + ((4, 3) if adjoint_a else (3, 4))
    b_shape = (5, 4) if adjoint_b else (4, 5)
    a = (rng.normal(size=a_shape) + 1j * rng.normal(size=a_shape)).astype(dtype)
    b = (rng.normal(size=b_shape) + 1j * rng.normal(size=b_shape)).astype(dtype)
    expected_a = a.conj().swapaxes(-1, -2) if adjoint_a else a
    expected_b = b.conj().T if adjoint_b else b
    expected = 0.7 * (expected_a @ expected_b)
    a_backend, b_backend = xp.asarray(a), xp.asarray(b)
    namespace = array_namespace(a_backend)
    out = namespace.empty(expected.shape, dtype=getattr(namespace, dtype))
    result = matmul(a_backend, b_backend, alpha=0.7, adjoint_a=adjoint_a, adjoint_b=adjoint_b, out=out)
    tolerance = 2e-6 if dtype == 'complex64' else 2e-14
    np.testing.assert_allclose(to_numpy(result), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(to_numpy(a_backend), a)
    np.testing.assert_array_equal(to_numpy(b_backend), b)
    assert result.dtype == a_backend.dtype
