import numpy as np
import pytest

import hcipy._math.buffer_pool as pool


def test_empty_shape_and_dtype():
    with pool.empty((3, 4), np.float64) as array:
        assert array.shape == (3, 4)
        assert array.dtype == np.float64


def test_empty_integer_shape_is_1d():
    with pool.empty(5, np.complex128) as array:
        assert array.shape == (5,)
        assert array.dtype == np.complex128


def test_zeros_shape_dtype_and_contents():
    with pool.zeros((3, 4), np.float64) as array:
        assert array.shape == (3, 4)
        assert array.dtype == np.float64
        assert np.all(array == 0)


def test_ones_shape_dtype_and_contents():
    with pool.ones((3, 4), np.float64) as array:
        assert array.shape == (3, 4)
        assert array.dtype == np.float64
        assert np.all(array == 1)


def test_empty_rejects_object_dtype():
    with pytest.raises(TypeError):
        with pool.empty((3,), object):
            pass
