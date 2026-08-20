from hcipy import *
import numpy as np

def test_micro_lens_array_default():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = make_pupil_grid(11, 1)
    focal_length = 1.0

    mla = MicroLensArray(input_grid, lenslet_grid, focal_length)

    assert mla.input_grid is input_grid
    assert mla.focal_length == focal_length
    assert mla.mla_grid is lenslet_grid
    assert mla.mla_index.size == input_grid.size
    assert mla.mla_opd.size == input_grid.size
    assert np.all(mla.mla_index >= 0)
    assert np.all(mla.mla_index < lenslet_grid.size)
    assert np.all(mla.mla_opd <= 0)

    wf = Wavefront(input_grid.ones())
    wf_out = mla.forward(wf)
    assert isinstance(wf_out, Wavefront)

    wf_back = mla.backward(wf_out)
    assert isinstance(wf_back, Wavefront)
    assert np.allclose(wf.electric_field, wf_back.electric_field)

def test_micro_lens_array_with_shape():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = CartesianGrid(SeparatedCoords(([0.0], [0.0])))
    focal_length = 1.0
    lenslet_shape = make_circular_aperture(0.1)

    mla = MicroLensArray(input_grid, lenslet_grid, focal_length, lenslet_shape=lenslet_shape)

    assert mla.mla_index.size == input_grid.size
    assert mla.mla_opd.size == input_grid.size

    inside = lenslet_shape(input_grid) != 0
    assert np.all(mla.mla_index[inside] == 0)
    assert np.all(mla.mla_index[~inside] == -1)
    assert np.all(mla.mla_opd[inside] < 0)
    assert np.all(mla.mla_opd[~inside] == 0)

    wf = Wavefront(input_grid.ones())
    wf_out = mla.forward(wf)
    assert isinstance(wf_out, Wavefront)

    wf_back = mla.backward(wf_out)
    assert isinstance(wf_back, Wavefront)
    assert np.allclose(wf.electric_field, wf_back.electric_field)

def test_micro_lens_array_multiple_lenslets():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = make_pupil_grid(3, 0.6)
    focal_length = 1.0
    lenslet_shape = make_circular_aperture(0.15)

    mla = MicroLensArray(input_grid, lenslet_grid, focal_length, lenslet_shape=lenslet_shape)

    assert mla.mla_index.size == input_grid.size
    assigned = mla.mla_index >= 0
    assert np.any(assigned)
    assert np.all(mla.mla_index[assigned] < 9)
    assert np.all(mla.mla_opd[assigned] < 0)
    assert np.all(mla.mla_opd[~assigned] == 0)

    wf = Wavefront(input_grid.ones())
    wf_fwd = mla.forward(wf)
    wf_bwd = mla.backward(wf_fwd)

    # Make sure that no amplitude is applied.
    assert np.allclose(wf.electric_field, wf_bwd.electric_field)

def test_spherical_micro_lens_array():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = CartesianGrid(SeparatedCoords(([0.0], [0.0])))
    radius_of_curvature = 0.5
    lenslet_shape = make_circular_aperture(0.1)

    mla = SphericalMicroLensArray(input_grid, lenslet_grid, radius_of_curvature, lenslet_shape)

    assert mla.n == 1.5
    assert mla.radius_of_curvature == radius_of_curvature
    assert mla.mla_index.size == input_grid.size
    assert mla.mla_opd.size == input_grid.size
    assert hasattr(mla, 'surface_sag')

    inside = lenslet_shape(input_grid) != 0
    assert np.all(mla.mla_index[inside] == 0)
    assert np.all(mla.mla_index[~inside] == -1)
    assert np.all(mla.mla_opd[inside] != 0)
    assert np.all(mla.mla_opd[~inside] == 0)

    wf = Wavefront(input_grid.ones())
    wf_out = mla.forward(wf)
    assert isinstance(wf_out, Wavefront)

    wf_back = mla.backward(wf_out)
    assert isinstance(wf_back, Wavefront)
    assert np.allclose(wf.electric_field, wf_back.electric_field)

def test_spherical_micro_lens_array_refractive_index():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = CartesianGrid(SeparatedCoords(([0.0], [0.0])))
    radius_of_curvature = 0.5
    lenslet_shape = make_circular_aperture(0.1)

    mla = SphericalMicroLensArray(input_grid, lenslet_grid, radius_of_curvature, lenslet_shape, refractive_index=2.0)
    assert mla.n == 2.0

def test_even_asphere_micro_lens_array():
    input_grid = make_pupil_grid(64, 1)
    lenslet_grid = CartesianGrid(SeparatedCoords(([0.0], [0.0])))
    radius_of_curvature = 0.5
    lenslet_shape = make_circular_aperture(0.1)

    mla = EvenAsphereMicroLensArray(input_grid, lenslet_grid, radius_of_curvature, lenslet_shape)

    assert mla.conic_constant == 0
    assert mla.aspheric_coefficients == []
    assert mla.n == 1.5
    assert mla.radius_of_curvature == radius_of_curvature
    assert mla.mla_index.size == input_grid.size

    mla2 = EvenAsphereMicroLensArray(input_grid, lenslet_grid, radius_of_curvature, lenslet_shape,
                                      refractive_index=1.7, conic_constant=-0.5,
                                      aspheric_coefficients=[1e-5, 1e-6])

    assert mla2.conic_constant == -0.5
    assert mla2.aspheric_coefficients == [1e-5, 1e-6]
    assert mla2.n == 1.7

    inside = lenslet_shape(input_grid) != 0
    assert np.all(mla2.mla_index[inside] == 0)
    assert np.all(mla2.mla_index[~inside] == -1)
    assert np.all(mla2.mla_opd[inside] != 0)
    assert np.all(mla2.mla_opd[~inside] == 0)

    wf = Wavefront(input_grid.ones())
    wf_out = mla2.forward(wf)
    assert isinstance(wf_out, Wavefront)

    wf_back = mla2.backward(wf_out)
    assert isinstance(wf_back, Wavefront)
    assert np.allclose(wf.electric_field, wf_back.electric_field)
