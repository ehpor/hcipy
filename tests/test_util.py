import numpy as np
import os
import pytest
from hcipy import *

def test_grid_io():
    grid1 = make_pupil_grid(128)
    grid2 = CartesianGrid(SeparatedCoords(grid1.separated_coords))
    grid3 = grid1.as_('polar')

    grids = [grid1, grid2, grid3]
    formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
    filenames = ['grid_test.' + fmt for fmt in formats]

    for g in grids:
        for fname in filenames:
            write_grid(g, fname)
            new_grid = read_grid(fname)

            assert hash(g) == hash(new_grid)

            os.remove(fname)

def test_field_io():
    grid = make_pupil_grid(128)
    field = make_circular_aperture(1)(grid)

    formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
    filenames = ['field_test.' + fmt for fmt in formats]

    for fname in filenames:
        write_field(field, fname)
        new_field = read_field(fname)

        assert np.allclose(field, new_field)
        assert hash(field.grid) == hash(new_field.grid)

        os.remove(fname)

def test_mode_basis_io():
    grid = make_pupil_grid(128)
    mode_bases = [
        make_zernike_basis(20, 1, grid, 1),
        make_xinetics_influence_functions(grid, 8, 1 / 8)
    ]

    formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
    filenames = ['mode_basis_test.' + fmt for fmt in formats]

    for mode_basis in mode_bases:
        for fname in filenames:
            write_mode_basis(mode_basis, fname)
            new_mode_basis = read_mode_basis(fname)

            assert hash(new_mode_basis.grid) == hash(mode_basis.grid)
            assert new_mode_basis.is_sparse == mode_basis.is_sparse

            for i in range(mode_basis.num_modes):
                assert np.allclose(mode_basis[i], new_mode_basis[i])

            os.remove(fname)


def test_finite_difference():

    grid = make_pupil_grid(16)
    surface = grid.ones()

    Dx = make_derivative_matrix(grid, axis='x')
    Dy = make_derivative_matrix(grid, axis='y')

    surface_dx = Dx.dot(surface)
    surface_dy = Dy.dot(surface)

    assert abs(np.median(surface_dx)) < 1e-10
    assert abs(np.median(surface_dy)) < 1e-10

    laplacian = make_laplacian_matrix(grid)
    surface_lap = laplacian.dot(surface)
    assert abs(np.median(surface_lap)) < 1e-10

    # Test if numpy array is handled correctly b
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_operator = generate_convolution_matrix(grid, laplacian_kernel)
    surface_lap = laplacian_operator.dot(surface)
    assert abs(np.median(surface_lap)) < 1e-10

def test_poisson():
    num_trials = 100
    num_runs = 100000
    lam = 100.0
    sigma = np.sqrt(lam / num_trials)

    lam_realization = large_poisson(lam * np.ones((num_trials, num_runs)), thresh=1e6)

    assert (np.std(np.mean(lam_realization, axis=0) - lam) - sigma) / sigma < 1e-2

    num_trials = 100
    num_runs = 100000
    lam = 10000000.0
    sigma = np.sqrt(lam / num_trials)

    lam_realization = large_poisson(lam * np.ones((num_trials, num_runs)), thresh=1e6)

    assert (np.std(np.mean(lam_realization, axis=0) - lam) - sigma) / sigma < 1e-2

def test_gamma():
    num_trials = 100
    num_runs = 100000
    lam = 100.0
    theta = 1 / 10.0
    mean = lam * theta
    sigma = np.sqrt(lam * theta**2 / num_trials)

    lam_realization = large_gamma(lam * np.ones((num_trials, num_runs)), theta, thresh=1e6)

    assert (np.std(np.mean(lam_realization, axis=0) - mean) - sigma) / sigma < 1e-2

    num_trials = 100
    num_runs = 100000
    lam = 10000000.0
    theta = 1 / 10.0

    mean = lam * theta
    sigma = np.sqrt(lam * theta**2 / num_trials)

    lam_realization = large_gamma(lam * np.ones((num_trials, num_runs)), theta, thresh=1e6)

    assert (np.std(np.mean(lam_realization, axis=0) - mean) - sigma) / sigma < 1e-2

def test_emccd_noise():
    photo_electron_flux = 1000.0
    read_noise = 0
    emgain = 500

    num_trials = 100
    num_runs = 100000

    sigma = np.sqrt(2 * photo_electron_flux / num_trials)
    noise = make_emccd_noise(photo_electron_flux * np.ones((num_trials, num_runs)), read_noise, emgain)

    assert abs(np.std(np.mean(noise, axis=0) / emgain - photo_electron_flux) - sigma) / sigma < 1e-2

def test_state_space_dynamics_stationary_distribution():
    ar_coeffs = np.array([0.5, -0.3])
    noise_variance = 0.01
    dt = 0.01
    A, B, C = ar_to_state_space(ar_coeffs, noise_variance, dt)

    samples = []
    for i in range(1000):
        dynamics = StateSpaceDynamics(A, B, C, seed=i)
        samples.append(dynamics.state)

    samples = np.array(samples)
    empirical_cov = np.cov(samples.T)

    dynamics = StateSpaceDynamics(A, B, C)
    theoretical_cov = dynamics.stationary_covariance

    relative_error = np.linalg.norm(empirical_cov - theoretical_cov) / np.linalg.norm(theoretical_cov)
    assert relative_error < 0.1

def test_state_space_dynamics_explicit_initial_state():
    ar_coeffs = np.array([0.5, -0.3])
    noise_variance = 0.01
    dt = 0.01
    A, B, C = ar_to_state_space(ar_coeffs, noise_variance, dt)

    initial_state = np.array([0.1, 0.2])
    dynamics = StateSpaceDynamics(A, B, C, initial_state=initial_state, seed=42)

    assert np.allclose(dynamics.state, initial_state)

def test_state_space_dynamics_backwards_evolution():
    ar_coeffs = np.array([0.5, -0.3])
    noise_variance = 0.01
    dt = 0.01
    A, B, C = ar_to_state_space(ar_coeffs, noise_variance, dt)

    dynamics = StateSpaceDynamics(A, B, C, seed=42)
    dynamics.evolve_until(1.0)

    with pytest.raises(ValueError):
        dynamics.evolve_until(0.5)

def test_state_space_dynamics_long_term_stationarity():
    ar_coeffs = np.array([0.5, -0.3])
    noise_variance = 0.01
    dt = 0.01
    A, B, C = ar_to_state_space(ar_coeffs, noise_variance, dt)

    samples = []
    for i in range(1000):
        dynamics = StateSpaceDynamics(A, B, C, seed=i)
        dynamics.evolve_until(10.0)
        samples.append(dynamics.state)

    samples = np.array(samples)
    empirical_cov = np.cov(samples.T)

    dynamics = StateSpaceDynamics(A, B, C)
    theoretical_cov = dynamics.stationary_covariance

    relative_error = np.linalg.norm(empirical_cov - theoretical_cov) / np.linalg.norm(theoretical_cov)
    assert relative_error < 0.1

def test_ar_to_state_space():
    ar_coeffs = np.array([0.5, -0.3])
    noise_variance = 0.01
    dt = 0.01
    A, B, C = ar_to_state_space(ar_coeffs, noise_variance, dt)

    # Check stability (all eigenvalues should have negative real parts)
    eigenvalues = np.linalg.eigvals(A)
    assert all(np.real(eigenvalues) < 0)

    # Check shapes
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)

    # Check that C selects first state
    assert np.allclose(C, np.array([[1, 0]]))

    # Check that noise goes to first state
    assert B[0, 0] > 0
    assert B[1, 0] == 0

def test_make_random_state_space():
    A = make_random_state_space(n_states=4, bandwidth=2.0, seed=42)

    eigenvalues = np.linalg.eigvals(A)
    assert all(np.real(eigenvalues) < 0)

def test_make_continuous_time_companion():
    poles = np.array([-0.5, -0.3])
    A, C = make_continuous_time_companion_matrix(poles)

    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_sorted = np.sort(np.real(eigenvalues))
    poles_sorted = np.sort(poles)

    assert np.allclose(eigenvalues_sorted, poles_sorted)
    assert C.shape == (1, 2)
    assert np.allclose(C, np.array([[1, 0]]))
