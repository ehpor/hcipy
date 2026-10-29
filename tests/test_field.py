from hcipy import *
import hcipy
import numpy as np
import copy
import pytest
import pickle
from hcipy._math.random import make_random_generator
from hcipy._math.backends import all_close, array_namespace, to_numpy

if Configuration().use_array_api:
    import array_api_strict as xp
else:
    import numpy as xp

def test_field_dot():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    a = rng.normal(size=(3, grid.size))
    A = rng.normal(size=(3, 3, grid.size))

    a = Field(a, grid)
    A = Field(A, grid)

    b = field_dot(A, a)
    bb = xp.stack([xp.matmul(A[..., i].data, a[..., i].data) for i in range(grid.size)]).T
    bb = Field(bb, grid)

    assert all_close(b, bb)

    b = field_dot(a, a)
    bb = xp.stack([xp.matmul(a[..., i].data, a[..., i].data) for i in range(grid.size)])
    bb = Field(bb, grid)

    assert all_close(b, bb)

    B = field_dot(A, A)
    BB = xp.empty_like(B.data)
    for i in range(grid.size):
        BB[..., i] = xp.matmul(A[..., i].data, A[..., i].data)

    assert all_close(B, BB)

    b = field_dot(a, a)
    bb = xp.stack([xp.matmul(a[..., i].data, a[..., i].data) for i in range(grid.size)])

    assert all_close(b, bb)

    n = rng.normal(size=3)

    b = field_dot(A, n)
    bb = xp.stack([xp.matmul(A[..., i].data, n) for i in range(grid.size)]).T

    assert all_close(b, bb)

    b = field_dot(n, A)
    bb = xp.stack([xp.matmul(n, A[..., i].data) for i in range(grid.size)]).T

    assert all_close(b, bb)

    N = rng.normal(size=(3, 3))

    B = field_dot(A, N)
    BB = xp.empty_like(B.data)
    for i in range(grid.size):
        BB[..., i] = xp.matmul(A[..., i].data, N)

    assert all_close(B, BB)

def test_field_trace():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    B = field_trace(A)
    BB = xp.stack([xp.linalg.trace(A[..., i].data) for i in range(grid.size)])

    assert all_close(B, BB)

def test_field_dot_singleton_array():
    grid = make_pupil_grid(2, xp=xp)
    b = Field(xp.ones((1, grid.size)), grid)
    a = xp.asarray([2.0])

    result = field_dot(a, b)

    assert result.shape == (grid.size,)

def test_field_inv():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    B = field_inv(A)
    BB = xp.stack([xp.linalg.inv(A[..., i].data) for i in range(grid.size)], axis=-1)

    assert all_close(B, BB)

def test_field_transpose():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    B = field_transpose(A)
    BB = xp.stack([xp.permute_dims(A[..., i].data, (1, 0)) for i in range(grid.size)], axis=-1)

    assert all_close(B, BB)

def test_field_conjugate_transpose():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    B = field_conjugate_transpose(A)
    BB = xp.stack([xp.permute_dims(xp.conj(A[..., i].data), (1, 0)) for i in range(grid.size)], axis=-1)

    assert all_close(B, BB)

def test_field_adjoint():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    B = field_adjoint(A)
    BB = xp.stack([xp.linalg.inv(A[..., i].data) * xp.linalg.det(A[..., i].data) for i in range(grid.size)], axis=-1)

    assert all_close(B, BB)

def test_field_inverse_tikhonov():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    for reg in [1e-1, 1e-3, 1e-6]:
        B = field_inverse_tikhonov(A, reg)
        BB = xp.stack([xp.asarray(inverse_tikhonov(A[..., i].data, reg)) for i in range(grid.size)], axis=-1)

        assert all_close(B, BB)

def test_field_inverse_truncated():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    for reg in [1e-1, 1e-3, 1e-6]:
        B = field_inverse_truncated(A, reg)
        BB = xp.stack([xp.asarray(inverse_truncated(A[..., i].data, reg)) for i in range(grid.size)], axis=-1)

        assert all_close(B, BB)

def test_field_inverse_truncated_modal():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, 3, grid.size)), grid)

    for num_modes in [1, 2]:
        B = field_inverse_truncated_modal(A, num_modes)
        BB = xp.stack([xp.asarray(inverse_truncated_modal(xp.asarray(A[..., i].data), num_modes)) for i in range(grid.size)], axis=-1)

        assert all_close(B, BB)

def test_field_cross():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(3, grid.size)), grid)
    B = Field(rng.normal(size=(3, grid.size)), grid)

    C = field_cross(A, B)
    CC = xp.stack([xp.linalg.cross(A[..., i].data, B[..., i].data) for i in range(grid.size)], axis=-1)

    assert all_close(C, CC)

def test_field_svd():
    grid = make_pupil_grid(2, xp=xp)

    rng = make_random_generator(xp)
    A = Field(rng.normal(size=(5, 10, grid.size)), grid)

    U, S, Vh = field_svd(A)
    u, s, vh = field_svd(A, False)

    for i in range(grid.size):
        svd = xp.linalg.svd(A[..., i].data)

        assert all_close(U[..., i], svd[0])
        assert all_close(S[..., i], svd[1])
        assert all_close(Vh[..., i], svd[2])

        svd2 = xp.linalg.svd(A[..., i].data, full_matrices=False)

        assert all_close(u[..., i], svd2[0])
        assert all_close(s[..., i], svd2[1])
        assert all_close(vh[..., i], svd2[2])

def test_grid_hashing_and_comparison():
    grid1 = make_pupil_grid(128, xp=xp)

    grid2 = CartesianGrid(SeparatedCoords(copy.deepcopy(grid1.separated_coords)))
    assert hash(grid1) != hash(grid2)
    assert grid1 != grid2
    assert grid2 != grid1

    grid3 = CartesianGrid(UnstructuredCoords(copy.deepcopy(grid1.coords)))
    assert hash(grid1) != hash(grid3)
    assert grid1 != grid3
    assert grid3 != grid1
    assert grid2 != grid3
    assert grid3 != grid2

    grid4 = make_pupil_grid(128, xp=xp)
    print('start')
    assert hash(grid1) == hash(grid4)
    assert grid1 == grid4

    grid5 = PolarGrid(grid1.coords)
    assert hash(grid1) != hash(grid5)
    assert grid1 != grid5
    assert grid5 != grid1

    grid6 = CartesianGrid(copy.deepcopy(grid1.coords))
    assert hash(grid1) == hash(grid6)
    assert grid1 == grid6

    grid7 = grid1.scaled(2)
    assert hash(grid1) != hash(grid7)
    assert grid1 != grid7

    grid8 = grid1.scaled(2)
    assert hash(grid1) != hash(grid8)
    assert hash(grid7) == hash(grid8)
    assert grid1 != grid8
    assert grid7 == grid8

    grid9 = make_pupil_grid(256, xp=xp)
    assert hash(grid1) != hash(grid9)
    assert grid1 != grid9

    grid10 = CartesianGrid(SeparatedCoords(copy.deepcopy(grid2.separated_coords)))
    assert hash(grid2) == hash(grid10)
    assert grid2 == grid10

    assert grid1 != 0
    assert grid1 != 'string'

def test_grid_supersampled():
    g = make_uniform_grid(128, [1, 1], xp=xp)
    g2 = make_supersampled_grid(g, 4)
    g3 = make_subsampled_grid(g2, 4)

    assert all_close(g.x, g3.x)
    assert all_close(g.y, g3.y)

    g4 = make_supersampled_grid(make_supersampled_grid(g, 2), 2)

    assert all_close(g2.x, g4.x)
    assert all_close(g2.y, g4.y)

def allclose(x1, x2, /, *, rtol=1e-5, atol=1e-8):
    xp = x1.__array_namespace__()

    try:
        diff = xp.abs(x1 - x2)
    except Exception:
        return False

    tolerance = atol + rtol * xp.abs(x2)
    close = diff <= tolerance

    return xp.all(close)

@pytest.mark.parametrize('Field', [hcipy.field.NewStyleField, hcipy.field.OldStyleField])
def test_field_arithmetic(Field):
    grid = make_pupil_grid(16, xp=xp)

    M = np.random.randn(grid.size, grid.size)

    a_data = np.ones(grid.size)
    b_data = np.ones(grid.size)

    a = Field(a_data, grid)
    b = Field(b_data, grid)

    fxps = a.__array_namespace__()

    assert allclose(a, b)
    assert allclose(a + b, 2)
    assert allclose(a - b, 0)
    assert allclose(a * b, a)

    assert is_field(a + b)
    assert is_field(a - b)
    assert is_field(a * b)
    assert is_field(fxps.exp(2j * a))

    assert is_field(a.conj())
    assert is_field(a.conjugate())
    assert is_field(a.clip(-1, 1))
    assert is_field(a.repeat(10))

    assert a.size == a.grid.size
    assert is_field(a.astype('bool'))
    assert allclose(a.sum(), a_data.sum())

    a[0] = 6
    a[1:2] = 3

    assert a[0] == 6
    assert a[1] == 3

    assert not is_field(a_data)

    assert not is_field(M.dot(a_data))

    assert allclose(a.imag, 0)

@pytest.mark.parametrize('Field', [hcipy.field.NewStyleField, hcipy.field.OldStyleField])
def test_field_pickle(Field):
    grid = make_pupil_grid(16, xp=xp)

    a = Field(np.ones(grid.size), grid)

    state = pickle.dumps(a)
    b = pickle.loads(state)

    assert allclose(a, b)
    assert a.grid == b.grid


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('tensor_shape', [(), (2,), (2, 3)])
def test_new_style_field_dict_serialization(xp, dtype, tensor_shape):
    xp = array_namespace(xp.asarray(0.0))
    coords = RegularCoords(delta=(0.25, 0.5), dims=(4, 3), zero=(-0.75, 1), xp=xp)
    weights = np.arange(1, 13, dtype=np.float64) / 8
    grid = CartesianGrid(coords, weights=xp.asarray(weights))

    shape = tensor_shape + (grid.size,)
    expected = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if np.issubdtype(expected.dtype, np.complexfloating):
        expected += 1j * (expected.real + 1)
    data = xp.asarray(expected)
    field = hcipy.field.NewStyleField(data, grid)

    tree = field.to_dict()

    assert type(tree['values']) is np.ndarray
    assert tree['values'].dtype == expected.dtype
    assert tree['xp_name'] == xp.__name__
    np.testing.assert_array_equal(tree['values'], expected)
    assert tree['grid']['coordinate_system'] == 'cartesian'
    assert tree['grid']['coords'] == {'type': 'regular', 'delta': [0.25, 0.5], 'dims': (4, 3), 'zero': [-0.75, 1.0], 'xp_name': xp.__name__}

    restored = hcipy.field.NewStyleField.from_dict(tree)

    assert array_namespace(restored.data).__name__ == xp.__name__
    assert restored.data.dtype == data.dtype
    np.testing.assert_array_equal(to_numpy(restored.data), expected)
    assert restored.tensor_shape == tensor_shape
    assert restored.grid == grid
    assert restored.grid.shape == (3, 4)
    assert restored.grid.coords._xp is xp
    np.testing.assert_array_equal(to_numpy(restored.grid.weights), weights)

    assert field.data is data
    assert field.grid is grid


def test_new_style_field_dict_serialization_legacy_defaults_to_numpy():
    grid = make_pupil_grid([4, 3])
    values = np.arange(grid.size, dtype=np.complex128) * (1 + 2j)
    tree = {
        'values': values,
        'grid': grid.to_dict(),
    }

    restored = hcipy.field.NewStyleField.from_dict(tree)

    assert type(restored.data) is np.ndarray
    assert restored.data.dtype == values.dtype
    np.testing.assert_array_equal(restored.data, values)
    assert restored.grid == grid


def test_old_style_field_dict_serialization():
    grid = make_pupil_grid([4, 3])
    values = np.arange(grid.size, dtype=np.complex128) * (1 + 2j)
    field = hcipy.field.OldStyleField(values, grid)

    tree = field.to_dict()
    restored = hcipy.field.OldStyleField.from_dict(tree)

    assert tree['values'].dtype == values.dtype
    assert restored.dtype == values.dtype
    np.testing.assert_array_equal(tree['values'], values)
    np.testing.assert_array_equal(restored, values)
    assert restored.grid == grid
