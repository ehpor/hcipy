import numpy as np

from hcipy import *


def test_single_segment_mode_matches_basis():
    pupil_grid = make_pupil_grid(64)
    segment_flat_to_flat = 1
    num_modes = 4

    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=0,
        segment_flat_to_flat=segment_flat_to_flat,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=num_modes
    )

    surface.set_segment_coefficients(0, {2: 1}, indexing='noll')

    segment_circum_diameter = segment_flat_to_flat * 2 / np.sqrt(3)
    expected_basis = make_hexike_basis(pupil_grid, num_modes, segment_circum_diameter, hexagon_angle=np.pi / 2)

    assert np.allclose(surface.surface, expected_basis[1])


def test_phase_scales_with_wavelength():
    pupil_grid = make_pupil_grid(32)
    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=0,
        segment_flat_to_flat=1,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=2
    )

    surface.set_segment_coefficients(0, {1: 10e-9}, indexing='noll')

    wavelength = 500e-9
    phase1 = surface.phase_for(wavelength)
    phase2 = surface.phase_for(2 * wavelength)

    assert np.allclose(phase2, 0.5 * phase1)


def test_forward_backward_roundtrip():
    pupil_grid = make_pupil_grid(32)
    aperture = make_circular_aperture(1)(pupil_grid)
    wavelength = 500e-9

    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=0,
        segment_flat_to_flat=1,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=2
    )
    surface.set_segment_coefficients(0, {1: 20e-9}, indexing='noll')

    wf = Wavefront(aperture, wavelength)
    wf_fwd = surface.forward(wf)
    wf_back = surface.backward(wf_fwd)

    assert np.allclose(wf_back.electric_field, wf.electric_field)


def test_zero_surface_is_noop():
    pupil_grid = make_pupil_grid(16)
    aperture = make_circular_aperture(1)(pupil_grid)
    wavelength = 500e-9

    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=0,
        segment_flat_to_flat=1,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=2
    )

    wf = Wavefront(aperture, wavelength)
    wf_out = surface.forward(wf)

    assert np.allclose(wf_out.electric_field, wf.electric_field)


def test_coefficient_interfaces():
    pupil_grid = make_pupil_grid(16)
    num_modes = 3

    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=1,
        segment_flat_to_flat=1,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=num_modes
    )

    num_segments = surface.coefficients.shape[0]

    # Array setter (2D)
    arr = np.arange(num_segments * num_modes).reshape(num_segments, num_modes) * 1e-9
    surface.coefficients = arr
    assert np.allclose(surface.coefficients, arr)

    # Flat setter
    surface.coefficients = arr.ravel()
    assert np.allclose(surface.coefficients, arr)

    # Dict setter
    surface.flatten()
    surface.set_segment_coefficients(0, {1: 1e-8, 3: 2e-8}, indexing='noll')
    assert np.isclose(surface.coefficients[0, 0], 1e-8)
    assert np.isclose(surface.coefficients[0, 2], 2e-8)

    # Dict-of-dicts
    surface.flatten()
    surface.set_coefficients_from_dict({0: {2: 5e-9}}, indexing='noll')
    assert np.isclose(surface.coefficients[0, 1], 5e-9)


def test_starting_ring_subsets_segments():
    pupil_grid = make_pupil_grid(16)

    surface = make_segment_hexike_surface_from_hex_aperture(
        num_rings=2,
        segment_flat_to_flat=1,
        gap_size=0,
        pupil_grid=pupil_grid,
        num_modes=1,
        starting_ring=1
    )

    # num_rings=2 has 1 + 6 + 12 = 19 segments; starting_ring=1 drops the center.
    assert surface.coefficients.shape[0] == 18
