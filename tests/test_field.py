from hcipy import *
import numpy as np
import copy

def test_field_dot():
	grid = make_pupil_grid(2)

	a = np.random.randn(3, grid.size)
	A = np.random.randn(3, 3, grid.size)

	a = Field(a, grid)
	A = Field(A, grid)

	b = field_dot(A, a)
	bb = np.array([A[...,i].dot(a[...,i]) for i in range(grid.size)]).T

	assert np.allclose(b, bb)

	b = field_dot(a, a)
	bb = np.array([a[...,i].dot(a[...,i]) for i in range(grid.size)]).T

	assert np.allclose(b, bb)

	B = field_dot(A, A)
	BB = np.empty_like(B)
	for i in range(grid.size):
		BB[...,i] = A[...,i].dot(A[...,i])
	
	assert np.allclose(B, BB)

	b = field_dot(a, a)
	bb = np.array([a[...,i].dot(a[...,i]) for i in range(grid.size)])

	assert np.allclose(b, bb)

	n = np.random.randn(3)

	b = field_dot(A, n)
	bb = np.array([A[...,i].dot(n) for i in range(grid.size)]).T

	assert np.allclose(b, bb)

	b = field_dot(n, A)
	bb = np.array([n.dot(A[...,i]) for i in range(grid.size)]).T

	assert np.allclose(b, bb)

	N = np.random.randn(3,3)

	B = field_dot(A, N)
	BB = np.empty_like(B)
	for i in range(grid.size):
		BB[...,i] = A[...,i].dot(N)
	
	assert np.allclose(B, BB)

def test_field_trace():
	grid = make_pupil_grid(2)
	
	A = Field(np.random.randn(3,3,grid.size), grid)
	
	B = field_trace(A)
	BB = np.array([np.trace(A[...,i]) for i in range(grid.size)])

	assert np.allclose(B, BB)

def test_field_inv():
	grid = make_pupil_grid(2)

	A = Field(np.random.randn(3,3,grid.size), grid)

	B = field_inv(A)
	BB = np.empty_like(B)
	for i in range(grid.size):
		BB[...,i] = np.linalg.inv(A[...,i])
	
	assert np.allclose(B, BB)

def test_field_inverse_tikhonov():
	grid = make_pupil_grid(2)

	A = Field(np.random.randn(3,3,grid.size), grid)

	for reg in [1e-1, 1e-3, 1e-6]:
		B = field_inverse_tikhonov(A, reg)
		BB = np.empty_like(B)
		for i in range(grid.size):
			BB[...,i] = inverse_tikhonov(A[...,i], reg)
		
		assert np.allclose(B, BB)

def test_field_svd():
	grid = make_pupil_grid(2)
	
	A = Field(np.random.randn(5,10,grid.size), grid)
	
	U, S, Vh = field_svd(A)
	u, s, vh = field_svd(A, False)

	for i in range(grid.size):
		svd = np.linalg.svd(A[...,i])

		assert np.allclose(U[...,i], svd[0])
		assert np.allclose(S[...,i], svd[1])
		assert np.allclose(Vh[...,i], svd[2])

		svd2 = np.linalg.svd(A[...,i], full_matrices=False)

		assert np.allclose(u[...,i], svd2[0])
		assert np.allclose(s[...,i], svd2[1])
		assert np.allclose(vh[...,i], svd2[2])

def test_grid_hashing():
	grid1 = make_pupil_grid(128)

	grid2 = CartesianGrid(SeparatedCoords(copy.deepcopy(grid1.separated_coords)))
	assert hash(grid1) != hash(grid2)

	grid3 = CartesianGrid(UnstructuredCoords(copy.deepcopy(grid1.coords)))
	assert hash(grid1) != hash(grid3)

	grid4 = make_pupil_grid(128)
	assert hash(grid1) == hash(grid4)

	grid5 = PolarGrid(grid1.coords)
	assert hash(grid1) != hash(grid5)

	grid6 = CartesianGrid(copy.deepcopy(grid1.coords))
	assert hash(grid1) == hash(grid6)

	grid7 = grid1.scaled(2)
	assert hash(grid1) != hash(grid7)

	grid8 = grid1.scaled(2)
	assert hash(grid1) != hash(grid8)
	assert hash(grid7) == hash(grid8)

	grid9 = make_pupil_grid(256)
	assert hash(grid1) != hash(grid9)
