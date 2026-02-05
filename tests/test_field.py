from hcipy import *
import hcipy
import numpy as np
import copy
import pytest
import pickle

def test_field_dot():
    grid = make_pupil_grid(2)

    a = np.random.randn(3, grid.size)
    A = np.random.randn(3, 3, grid.size)

    a = Field(a, grid)
    A = Field(A, grid)

    b = field_dot(A, a)
    bb = np.array([A[..., i].dot(a[..., i]) for i in range(grid.size)]).T

    assert np.allclose(b, bb)

    b = field_dot(a, a)
    bb = np.array([a[..., i].dot(a[..., i]) for i in range(grid.size)]).T

    assert np.allclose(b, bb)

    B = field_dot(A, A)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = A[..., i].dot(A[..., i])

    assert np.allclose(B, BB)

    b = field_dot(a, a)
    bb = np.array([a[..., i].dot(a[..., i]) for i in range(grid.size)])

    assert np.allclose(b, bb)

    n = np.random.randn(3)

    b = field_dot(A, n)
    bb = np.array([A[..., i].dot(n) for i in range(grid.size)]).T

    assert np.allclose(b, bb)

    b = field_dot(n, A)
    bb = np.array([n.dot(A[..., i]) for i in range(grid.size)]).T

    assert np.allclose(b, bb)

    N = np.random.randn(3, 3)

    B = field_dot(A, N)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = A[..., i].dot(N)

    assert np.allclose(B, BB)

def test_field_trace():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    B = field_trace(A)
    BB = np.array([np.trace(A[..., i]) for i in range(grid.size)])

    assert np.allclose(B, BB)

def test_field_inv():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    B = field_inv(A)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = np.linalg.inv(A[..., i])

    assert np.allclose(B, BB)

def test_field_transpose():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    B = field_transpose(A)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = A[..., i].T

    assert np.allclose(B, BB)

def test_field_conjugate_transpose():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    B = field_conjugate_transpose(A)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = A[..., i].T.conj()

    assert np.allclose(B, BB)

def test_field_adjoint():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    B = field_adjoint(A)
    BB = np.empty_like(B)
    for i in range(grid.size):
        BB[..., i] = np.linalg.inv(A[..., i]) * np.linalg.det(A[..., i])

    assert np.allclose(B, BB)

def test_field_inverse_tikhonov():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    for reg in [1e-1, 1e-3, 1e-6]:
        B = field_inverse_tikhonov(A, reg)
        BB = np.empty_like(B)

        for i in range(grid.size):
            BB[..., i] = inverse_tikhonov(A[..., i], reg)

        assert np.allclose(B, BB)

def test_field_inverse_truncated():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    for reg in [1e-1, 1e-3, 1e-6]:
        B = field_inverse_truncated(A, reg)
        BB = np.empty_like(B)

        for i in range(grid.size):
            BB[..., i] = inverse_truncated(A[..., i], reg)

        assert np.allclose(B, BB)

def test_field_inverse_truncated_modal():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, 3, grid.size), grid)

    for num_modes in [1, 2]:
        B = field_inverse_truncated_modal(A, num_modes)
        BB = np.empty_like(B)

        for i in range(grid.size):
            BB[..., i] = inverse_truncated_modal(np.asarray(A[..., i]), num_modes)

        assert np.allclose(B, BB)

def test_field_cross():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(3, grid.size), grid)
    B = Field(np.random.randn(3, grid.size), grid)

    C = field_cross(A, B)
    CC = np.empty_like(C)
    for i in range(grid.size):
        CC[..., i] = np.cross(A[:, i], B[:, i])

    assert np.allclose(C, CC)

def test_field_svd():
    grid = make_pupil_grid(2)

    A = Field(np.random.randn(5, 10, grid.size), grid)

    U, S, Vh = field_svd(A)
    u, s, vh = field_svd(A, False)

    for i in range(grid.size):
        svd = np.linalg.svd(A[..., i])

        assert np.allclose(U[..., i], svd[0])
        assert np.allclose(S[..., i], svd[1])
        assert np.allclose(Vh[..., i], svd[2])

        svd2 = np.linalg.svd(A[..., i], full_matrices=False)

        assert np.allclose(u[..., i], svd2[0])
        assert np.allclose(s[..., i], svd2[1])
        assert np.allclose(vh[..., i], svd2[2])

def test_grid_hashing_and_comparison():
    grid1 = make_pupil_grid(128)

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

    grid4 = make_pupil_grid(128)
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

    grid9 = make_pupil_grid(256)
    assert hash(grid1) != hash(grid9)
    assert grid1 != grid9

    grid10 = CartesianGrid(SeparatedCoords(copy.deepcopy(grid2.separated_coords)))
    assert hash(grid2) == hash(grid10)
    assert grid2 == grid10

    assert grid1 != 0
    assert grid1 != 'string'

def test_grid_supersampled():
    g = make_uniform_grid(128, [1, 1])
    g2 = make_supersampled_grid(g, 4)
    g3 = make_subsampled_grid(g2, 4)

    assert np.allclose(g.x, g3.x)
    assert np.allclose(g.y, g3.y)

    g4 = make_supersampled_grid(make_supersampled_grid(g, 2), 2)

    assert np.allclose(g2.x, g4.x)
    assert np.allclose(g2.y, g4.y)

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
    grid = make_pupil_grid(16)

    M = np.random.randn(grid.size, grid.size)

    a_data = np.ones(grid.size)
    b_data = np.ones(grid.size)

    a = Field(a_data, grid)
    b = Field(b_data, grid)

    xp = a.__array_namespace__()

    assert allclose(a, b)
    assert allclose(a + b, 2)
    assert allclose(a - b, 0)
    assert allclose(a * b, a)

    assert is_field(a + b)
    assert is_field(a - b)
    assert is_field(a * b)
    assert is_field(xp.exp(2j * a))

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
    grid = make_pupil_grid(16)

    a = Field(np.ones(grid.size), grid)

    state = pickle.dumps(a)
    b = pickle.loads(state)

    assert allclose(a, b)
    assert a.grid == b.grid

