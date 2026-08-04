from hcipy import *
import numpy as np
import scipy
import pytest

def test_zernike_modes():
    grid = make_pupil_grid(128)
    aperture_mask = make_circular_aperture(1)(grid) > 0

    modes = make_zernike_basis(200, 1, grid)

    assert np.abs(np.std(modes[0][aperture_mask])) < 2e-2

    for m in modes[1:]:
        assert np.abs(np.std(m[aperture_mask]) - 1) < 2e-2

    for i, m in enumerate(modes):
        zn, zm = noll_to_zernike(i + 1)
        assert np.allclose(m, zernike(zn, zm, grid=grid))

def test_zernike_indices():
    for i in range(1, 500):
        n, m = noll_to_zernike(i)
        assert i == zernike_to_noll(n, m)

        n, m = ansi_to_zernike(i)
        assert i == zernike_to_ansi(n, m)

def test_hexike_modes():
    grid = make_pupil_grid(128)
    aperture_mask = make_hexagonal_aperture(1)(grid) > 0

    num_modes = 2 + 3 + 4 + 5 + 6
    modes = make_hexike_basis(grid, num_modes, 1)

    assert abs(np.std(modes[0][aperture_mask])) < 2e-2

    for m in modes[1:]:
        assert abs(np.std(m[aperture_mask]) - 1) < 2e-2

    for i, m in enumerate(modes):
        zn, zm = noll_to_zernike(i + 1)
        assert np.allclose(m, hexike(zn, zm, 1, grid=grid))

def test_zernike_ansi_noll():
    grid = make_pupil_grid(64)

    for index in [1, 4, 5, 6]:
        mode_ansi = zernike_ansi(index)(grid)

        n, m = ansi_to_zernike(index)
        mode_nm = zernike(n, m)(grid)

        assert np.allclose(mode_ansi, mode_nm)

        mode_noll = zernike_noll(index)(grid)

        n, m = noll_to_zernike(index)
        mode_nm = zernike(n, m)(grid)

        assert np.allclose(mode_noll, mode_nm)

def test_hexike_ansi_noll():
    grid = make_pupil_grid(64)

    for index in [1, 4, 5, 6]:
        mode_ansi = hexike_ansi(index, 1)(grid)

        n, m = ansi_to_zernike(index)
        mode_nm = hexike(n, m, 1)(grid)

        assert np.allclose(mode_ansi, mode_nm)

        mode_noll = hexike_noll(index, 1)(grid)

        n, m = noll_to_zernike(index)
        mode_nm = hexike(n, m, 1)(grid)

        assert np.allclose(mode_noll, mode_nm)

def test_hexike_cache():
    grid = make_pupil_grid(64)
    circum_diameter = 1
    n = 2
    m = 2

    cache = {}
    basis = make_hexike_basis(grid, 20, circum_diameter, cache=cache)

    m1 = hexike(n, m, circum_diameter, grid=grid, cache=cache)
    m2 = basis[zernike_to_noll(n, m) - 1]

    assert np.allclose(m1, m2)

def test_zernike_cache():
    grid = make_pupil_grid(64)
    circum_diameter = 1
    n = 2
    m = 2

    cache = {}
    basis = make_zernike_basis(20, circum_diameter, grid, cache=cache)

    m1 = zernike(n, m, circum_diameter, grid=grid, cache=cache)
    m2 = basis[zernike_to_noll(n, m) - 1]

    assert np.allclose(m1, m2)

@pytest.mark.parametrize('bc', ['dirichlet', 'neumann'])
def test_disk_harmonic_modes(bc):
    grid = make_pupil_grid(128)
    aperture_mask = make_circular_aperture(1)(grid) > 0

    num_modes = 20

    modes = make_disk_harmonic_basis(grid, num_modes, bc=bc)

    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            product = np.sum((m1 * m2)[aperture_mask])
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-2

def test_lp_modes():
    grid = make_pupil_grid(128)

    # Test for single-mode
    modes = make_lp_modes(grid, 2.4, 0.1, return_betas=False)
    assert len(modes) == 1

    # Test orthogonality
    modes = make_lp_modes(grid, 25, 0.1, return_betas=False)
    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            product = np.real(np.sum(m1 * m2.conj() * grid.weights))
            assert np.abs(product - 1 if i == j else 0) <= 1e-2

def test_kirchhoff_love_basis():
    grid = make_pupil_grid(48)
    stiffness = make_circular_aperture(1)(grid)

    num_modes = 5
    modes, frequencies = make_kirchhoff_love_basis(stiffness, num_modes, return_frequencies=True)

    # The modes are orthonormal w.r.t. the mass-weighted inner product,
    # which is the area-weighted inner product for a uniform mass density.
    T = modes.transformation_matrix
    active = np.asarray(stiffness).ravel() > 0
    gram = (T[active] * np.sqrt(grid.weights)).T @ (T[active] * np.sqrt(grid.weights))

    assert np.allclose(gram, np.eye(num_modes), atol=1e-6)

    # The frequencies are positive, and sorted in ascending order.
    assert len(frequencies) == num_modes
    assert np.all(frequencies > 0)
    assert np.all(np.diff(frequencies) > 0)

def test_kirchhoff_love_basis_units():
    grid = make_pupil_grid(48)
    stiffness = make_circular_aperture(1)(grid)

    _, f = make_kirchhoff_love_basis(stiffness, 2, return_frequencies=True)

    # The frequency scales as sqrt(D / (rho h a^4)). Check the scaling
    # behavior of the returned frequencies, which are in Hz when the inputs
    # are in SI units.
    _, f_stiff = make_kirchhoff_love_basis(stiffness * 4, 2, return_frequencies=True)
    assert np.allclose(f_stiff / f, 2, rtol=1e-3)

    mass = Field(np.full(grid.size, 4.0), grid)
    _, f_mass = make_kirchhoff_love_basis(stiffness, 2, mass_density=mass, return_frequencies=True)
    assert np.allclose(f_mass / f, 0.5, rtol=1e-3)

    grid_small = make_pupil_grid(48, 0.5)
    stiffness_small = make_circular_aperture(0.5)(grid_small)
    _, f_small = make_kirchhoff_love_basis(stiffness_small, 2, return_frequencies=True)
    assert np.allclose(f_small / f, 4, rtol=1e-2)

def test_kirchhoff_love_basis_fixed_points():
    grid = make_pupil_grid(48)
    stiffness = make_circular_aperture(1)(grid)

    # Clamp the membrane at four points around the center.
    fp = np.zeros(grid.size)
    for p in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
        fp[grid.closest_to(p)] = 1
    fixed_points = Field(fp, grid)

    modes, frequencies = make_kirchhoff_love_basis(stiffness, 5, fixed_points=fixed_points, return_frequencies=True)

    # The modes are zero at the clamping points.
    for mode in modes:
        assert np.all(np.abs(np.asarray(mode)[fp > 0.5]) < 1e-10)

    # The clamping points kill the rigid-body modes, so all frequencies are
    # strictly positive.
    assert np.all(frequencies > 1e-8)

def test_kirchhoff_love_basis_errors():
    grid = make_pupil_grid(32)
    stiffness = make_circular_aperture(1)(grid)

    with pytest.raises(ValueError):
        make_kirchhoff_love_basis(stiffness.grid, 5)

    with pytest.raises(ValueError):
        make_kirchhoff_love_basis(stiffness, 5000)

    with pytest.raises(ValueError):
        make_kirchhoff_love_basis(stiffness, 5, mass_density=Field(np.zeros(grid.size), grid))

def test_sparse_mode_basis():
    transformation_matrix = np.empty((100 * 100, 100))
    for i in range(100):
        transformation_matrix[:, i] = np.random.randn(100 * 100)
        transformation_matrix[np.random.choice(100 * 100, 80 * 100, False), i] = 0

    mode_basis_1 = ModeBasis(transformation_matrix)

    assert mode_basis_1.is_dense
    assert not mode_basis_1.is_sparse
    assert not scipy.sparse.issparse(mode_basis_1.transformation_matrix)

    mode_basis_2 = mode_basis_1.to_dense(copy=False)

    assert mode_basis_2 is mode_basis_1

    mode_basis_3 = mode_basis_1.to_dense(copy=True)

    assert mode_basis_3 is not mode_basis_1
    assert np.allclose(mode_basis_1.transformation_matrix, mode_basis_3.transformation_matrix)

    mode_basis_4 = mode_basis_1.to_sparse()

    assert mode_basis_4.is_sparse
    assert not mode_basis_4.is_dense
    assert scipy.sparse.issparse(mode_basis_4.transformation_matrix)
    assert mode_basis_4.transformation_matrix.nnz < np.prod(mode_basis_4.transformation_matrix.shape)

    mode_basis_5 = mode_basis_4.to_dense()

    assert mode_basis_5.is_dense
    assert not mode_basis_5.is_sparse
    assert np.allclose(mode_basis_1.transformation_matrix, mode_basis_5.transformation_matrix)
    assert not scipy.sparse.issparse(mode_basis_5.transformation_matrix)

def test_gaussian_laguerre_modes():
    grid = make_focal_grid(32, 4)

    p_max = 5
    l_max = 5
    mode_field_diameter = 1

    modes = make_gaussian_laguerre_basis(grid, p_max, l_max, mode_field_diameter)
    num_modes = len(modes)

    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-6

def test_gaussian_hermite_modes():
    grid = make_focal_grid(32, 4)

    num_modes = 50
    mode_field_diameter = 1

    modes = make_gaussian_hermite_basis(grid, num_modes, mode_field_diameter)

    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-6

def test_fourier_modes():
    grid = make_pupil_grid(32)
    fourier_grid = make_fft_grid(grid, 1, 0.2)

    # Cosine modes
    cosine_modes = make_cosine_basis(grid, fourier_grid)
    num_modes = len(cosine_modes)

    for i, m1 in enumerate(cosine_modes):
        for j, m2 in enumerate(cosine_modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-12

    # Sine modes
    sine_modes = make_sine_basis(grid, fourier_grid)
    num_modes = len(sine_modes)

    for i, m1 in enumerate(sine_modes):
        for j, m2 in enumerate(sine_modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-12

    # Fourier basis
    fourier_modes = make_fourier_basis(grid, fourier_grid)
    num_modes = len(fourier_modes)

    for i, m1 in enumerate(fourier_modes):
        for j, m2 in enumerate(fourier_modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-12

    # Complex Fourier basis
    complex_fourier_modes = make_complex_fourier_basis(grid, fourier_grid)
    num_modes = len(complex_fourier_modes)

    for i, m1 in enumerate(complex_fourier_modes):
        for j, m2 in enumerate(complex_fourier_modes):
            product = np.sum((m1.conj() * m2) * grid.weights).real
            assert np.abs(product - np.eye(num_modes)[i, j]) < 1e-12
