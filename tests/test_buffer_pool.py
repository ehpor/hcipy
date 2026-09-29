import numpy as np
import pytest

import hcipy._math.buffer_pool as pool


def test_temp_shape_and_dtype():
    with pool.temp((3, 4), np.float64) as array:
        assert array.shape == (3, 4)
        assert array.dtype == np.float64


def test_temp_integer_shape_is_1d():
    with pool.temp(5, np.complex128) as array:
        assert array.shape == (5,)
        assert array.dtype == np.complex128


def test_temp_rejects_object_dtype():
    with pytest.raises(TypeError):
        with pool.temp((3,), object):
            pass
