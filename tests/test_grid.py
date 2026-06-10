from hcipy import *
from hcipy.field.coordinates import Coords
import pytest
from hcipy._math.backends import all_close
import math
if Configuration().core.use_new_style_fields:
    import array_api_strict as xp
else:
    import numpy as np

def test_unstructured_coords():
    coords = UnstructuredCoords([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], xp=xp)
    assert coords.size == 3
    assert len(coords) == 2
    assert xp.all(coords[0] == xp.asarray([1.0, 2.0, 3.0]))
    assert xp.all(coords[1] == xp.asarray([4.0, 5.0, 6.0]))

    coords_copy = coords.copy()
    assert coords == coords_copy

    coords_copy += 1
    assert coords != coords_copy
    assert all_close(coords_copy[0], xp.asarray([2.0, 3.0, 4.0]))
    assert all_close(coords_copy[1], xp.asarray([5.0, 6.0, 7.0]))

    coords_copy = coords.copy()
    coords_copy *= 2
    assert all_close(coords_copy[0], xp.asarray([2.0, 4.0, 6.0]))
    assert all_close(coords_copy[1], xp.asarray([8.0, 10.0, 12.0]))

    coords.reverse()
    assert xp.all(coords[0] == xp.asarray([3.0, 2.0, 1.0]))
    assert xp.all(coords[1] == xp.asarray([6.0, 5.0, 4.0]))

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_separated_coords():
    coords = SeparatedCoords([[1.0, 2.0, 3.0], [4.0, 5.0]], xp=xp)
    assert coords.size == 6
    assert len(coords) == 2
    assert coords.dims == (3, 2)
    assert coords.shape == (2, 3)
    assert xp.all(coords.separated_coords[0] == xp.asarray([1.0, 2.0, 3.0]))
    assert xp.all(coords.separated_coords[1] == xp.asarray([4.0, 5.0]))
    assert all_close(coords[0], xp.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
    assert all_close(coords[1], xp.asarray([4.0, 4.0, 4.0, 5.0, 5.0, 5.0]))

    coords_copy = coords.copy()
    assert coords == coords_copy

    coords_copy += (1, 2)
    assert coords != coords_copy
    assert all_close(coords_copy.separated_coords[0], xp.asarray([2.0, 3.0, 4.0]))
    assert all_close(coords_copy.separated_coords[1], xp.asarray([6.0, 7.0]))

    coords_copy = coords.copy()
    coords_copy *= 2
    assert all_close(coords_copy.separated_coords[0], xp.asarray([2.0, 4.0, 6.0]))
    assert all_close(coords_copy.separated_coords[1], xp.asarray([8.0, 10.0]))

    coords.reverse()
    assert xp.all(coords.separated_coords[0] == xp.asarray([3.0, 2.0, 1.0]))
    assert xp.all(coords.separated_coords[1] == xp.asarray([5.0, 4.0]))

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_regular_coords():
    coords = RegularCoords(delta=(0.5, 1), dims=(3, 2), zero=(0, 0), xp=xp)
    assert coords.size == 6
    assert len(coords) == 2
    assert coords.dims == (3, 2)
    assert coords.shape == (2, 3)
    assert tuple(coords.delta) == (0.5, 1)
    assert tuple(coords.zero) == (0, 0)

    assert all_close(coords.separated_coords[0], xp.asarray([0, 0.5, 1]))
    assert all_close(coords.separated_coords[1], xp.asarray([0.0, 1.0]))

    assert all_close(coords[0], xp.asarray([0, 0.5, 1, 0, 0.5, 1]))
    assert all_close(coords[1], xp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

    coords_copy = coords.copy()
    assert coords == coords_copy
    coords_copy += (1, 2)
    assert coords != coords_copy
    assert all_close(coords_copy.zero, xp.asarray([1.0, 2.0]))

    coords_copy = coords.copy()
    coords_copy *= 2
    assert all_close(coords_copy.delta, xp.asarray([1.0, 2.0]))
    assert all_close(coords_copy.zero, xp.asarray([0.0, 0.0]))

    coords.reverse()
    assert all_close(coords.delta, xp.asarray([-0.5, -1]))
    assert all_close(coords.zero, xp.asarray([1.0, 1.0]))

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_coords_arithmetic():
    # Unstructured
    c1 = UnstructuredCoords([[1.0, 2.0], [3.0, 4.0]], xp=xp)
    c2 = c1 + 1
    assert all_close(c2[0], xp.asarray([2.0, 3.0]))
    assert all_close(c2[1], xp.asarray([4.0, 5.0]))

    c2 = c1 - 1
    assert all_close(c2[0], xp.asarray([0.0, 1.0]))
    assert all_close(c2[1], xp.asarray([2.0, 3.0]))

    c2 = c1 * 2
    assert all_close(c2[0], xp.asarray([2.0, 4.0]))
    assert all_close(c2[1], xp.asarray([6.0, 8.0]))
    c2 = c1 / 2
    assert all_close(c2[0], xp.asarray([0.5, 1]))
    assert all_close(c2[1], xp.asarray([1.5, 2]))

    # Separated
    c1 = SeparatedCoords([[1.0, 2.0], [3.0, 4.0]], xp=xp)
    c2 = c1 + (1, 2)
    assert all_close(c2.separated_coords[0], xp.asarray([2.0, 3.0]))
    assert all_close(c2.separated_coords[1], xp.asarray([5.0, 6.0]))

    c2 = c1 - (1, 2)
    assert all_close(c2.separated_coords[0], xp.asarray([0.0, 1.0]))
    assert all_close(c2.separated_coords[1], xp.asarray([1.0, 2.0]))

    c2 = c1 * 2
    assert all_close(c2.separated_coords[0], xp.asarray([2.0, 4.0]))
    assert all_close(c2.separated_coords[1], xp.asarray([6.0, 8.0]))

    c2 = c1 / 2
    assert all_close(c2.separated_coords[0], xp.asarray([0.5, 1]))
    assert all_close(c2.separated_coords[1], xp.asarray([1.5, 2]))

    # Regular
    c1 = RegularCoords((1, 2), (3, 4), (5, 6), xp=xp)
    c2 = c1 + (1, 2)
    assert all_close(c2.zero, xp.asarray([6.0, 8.0]))

    c2 = c1 - (1, 2)
    assert all_close(c2.zero, xp.asarray([4.0, 4.0]))

    c2 = c1 * 2
    assert all_close(c2.delta, xp.asarray([2.0, 4.0]))
    assert all_close(c2.zero, xp.asarray([10.0, 12.0]))

    c2 = c1 / 2
    assert all_close(c2.delta, xp.asarray([0.5, 1]))
    assert all_close(c2.zero, xp.asarray([2.5, 3]))

def test_grid_creation():
    # Unstructured
    ug = Grid(UnstructuredCoords([[1, 2], [3, 4]], xp=xp))
    assert ug.is_unstructured
    assert not ug.is_separated
    assert not ug.is_regular
    assert ug.ndim == 2
    assert ug.size == 2

    # Separated
    sg = Grid(SeparatedCoords([[1, 2], [3, 4]], xp=xp))
    assert not sg.is_unstructured
    assert sg.is_separated
    assert not sg.is_regular
    assert sg.ndim == 2
    assert sg.size == 4
    assert sg.dims == (2, 2)
    assert sg.shape == (2, 2)

    # Regular
    rg = Grid(RegularCoords((1, 1), (2, 2), (0, 0), xp=xp))
    assert not rg.is_unstructured
    assert rg.is_separated
    assert rg.is_regular
    assert rg.ndim == 2
    assert rg.size == 4
    assert rg.dims == (2, 2)
    assert rg.shape == (2, 2)
    assert tuple(rg.delta) == (1, 1)
    assert tuple(rg.zero) == (0, 0)

def test_grid_exceptions():
    ug = Grid(UnstructuredCoords([[1, 2], [3, 4]], xp=xp))
    with pytest.raises(ValueError):
        ug.dims
    with pytest.raises(ValueError):
        ug.shape
    with pytest.raises(ValueError):
        ug.delta
    with pytest.raises(ValueError):
        ug.zero
    with pytest.raises(ValueError):
        ug.separated_coords
    with pytest.raises(ValueError):
        ug.regular_coords

    sg = Grid(SeparatedCoords([[1, 2], [3, 4]], xp=xp))
    with pytest.raises(ValueError):
        sg.delta
    with pytest.raises(ValueError):
        sg.zero
    with pytest.raises(ValueError):
        sg.regular_coords

def test_grid_subset():
    grid = CartesianGrid(RegularCoords((1, 1), (10, 10), (0, 0), xp=xp))
    subset_grid = grid.subset(grid.x > 3)

    assert subset_grid.size < grid.size

    field_xp = subset_grid.x.__array_namespace__()
    assert field_xp.all(subset_grid.x > 3)

    indices = field_xp.nonzero(grid.x > 3)[0]
    subset_grid2 = grid.subset(indices)
    assert subset_grid == subset_grid2

def test_cartesian_grid_transformations():
    grid = CartesianGrid(RegularCoords((1.0,), (8,), (0.0,), xp=xp))

    # Scaling
    sgrid = grid.scaled(2)
    assert all_close(sgrid.delta, xp.asarray([2.0]))
    sgrid.scale(0.5)
    assert all_close(sgrid.delta, xp.asarray([1.0]))
    assert grid == sgrid

    # Shifting
    sgrid = grid.shifted(2)
    assert all_close(sgrid.zero, xp.asarray([2.0]))
    sgrid.shift(-2)
    assert all_close(sgrid.zero, xp.asarray([0.0]))
    assert grid == sgrid

    # Rotation 2D
    grid2d = CartesianGrid(RegularCoords((1, 1), (3, 3), zero=(-1, -1), xp=xp))
    rgrid = grid2d.rotated(xp.pi / 2)
    assert all_close(rgrid.points, xp.asarray([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]]).T, atol=1e-4)

    # Rotation 3D (around the z-axis in this test)
    grid3d = CartesianGrid(RegularCoords((1, 1, 1), (2, 2, 2), (0, 0, 0), xp=xp))
    rgrid = grid3d.rotated(xp.pi / 2, axis=[0, 0, 1])
    assert all_close(rgrid.x, -grid3d.y)
    assert all_close(rgrid.y, grid3d.x)
    assert all_close(rgrid.z, grid3d.z)

def test_polar_grid_transformations():
    grid = PolarGrid(UnstructuredCoords([[1.0, 2.0], [xp.pi / 4, xp.pi / 2]], xp=xp))

    # Scaling
    sgrid = grid.scaled(2)
    assert all_close(sgrid.r, grid.r * 2)
    sgrid.scale(0.5)
    assert all_close(sgrid.r, grid.r)

    # Rotation
    rgrid = grid.rotated(xp.pi / 4)
    assert all_close(rgrid.theta, grid.theta + xp.pi / 4)
    rgrid.rotate(-xp.pi / 4)
    assert all_close(rgrid.theta, grid.theta)

    # Shifting
    sgrid = grid.shifted([1, 0])
    assert sgrid.is_('cartesian')
    # Check if the point (r=1, th=pi/4) which is (x=sqrt(2)/2, y=sqrt(2)/2) is shifted
    # to (x=1+sqrt(2)/2, y=sqrt(2)/2)
    assert all_close(sgrid.x[0], 1 + math.sqrt(2) / 2)
    assert all_close(sgrid.y[0], math.sqrt(2) / 2)

def test_coordinate_transformation():
    # 2D
    grid_cart = CartesianGrid(RegularCoords((1, 1), (3, 3), (0, 0), xp=xp))
    grid_pol = grid_cart.as_('polar')
    assert grid_pol.is_('polar')
    grid_cart2 = grid_pol.as_('cartesian')
    assert grid_cart2.is_('cartesian')
    assert all_close(grid_cart.points, grid_cart2.points, atol=1e-4)

def test_grid_reversal():
    grid = CartesianGrid(RegularCoords((1, 1), (10, 12), (0, 0), xp=xp))
    rev_grid = grid.reversed()
    assert not grid == rev_grid
    assert all_close(grid.points, xp.flip(rev_grid.points, (0,)))
    rev_grid.reverse()
    assert grid == rev_grid

def test_grid_closest_to():
    grid = CartesianGrid(RegularCoords((1, 1), (10, 10), (0, 0), xp=xp))
    p = [2.1, 3.8]
    idx = grid.closest_to(p)
    assert idx == 42  # Corresponds to point (2, 4).

def test_grid_field_creation():
    grid = CartesianGrid(RegularCoords((1,), (10,), (0,), xp=xp))
    f_zeros = grid.zeros()
    f_ones = grid.ones()
    f_empty = grid.empty()

    field_xp = f_zeros.__array_namespace__()

    assert field_xp.all(f_zeros == 0)
    assert field_xp.all(f_ones == 1)

    assert f_zeros.grid == grid
    assert f_ones.grid == grid
    assert f_empty.grid == grid

    assert f_zeros.shape == (10,)
    assert f_ones.shape == (10,)
    assert f_empty.shape == (10,)

    f_zeros_t = grid.zeros(tensor_shape=(2, 3))

    assert f_zeros_t.tensor_shape == (2, 3)
    assert f_zeros_t.shape == (2, 3, 10)

def test_grid_weights():
    # Regular grid
    grid = CartesianGrid(RegularCoords((0.1, 0.2), (10, 20), (0, 0), xp=xp))
    assert all_close(grid.weights, 0.1 * 0.2)

    # Separated grid
    x = xp.asarray([-1.0, 0.0, 1.0, 3.0])
    y = xp.asarray([10.0, 20.0])
    grid = CartesianGrid(SeparatedCoords([x, y], xp=xp))

    w_x = xp.concat((xp.asarray([x[1] - x[0]]), (x[2:] - x[:-2]) / 2, xp.asarray([x[-1] - x[-2]])))
    w_y = xp.concat((xp.asarray([y[1] - y[0]]), (y[2:] - y[:-2]) / 2, xp.asarray([y[-1] - y[-2]])))

    expected_weights = xp.outer(w_y, w_x).ravel()
    assert all_close(grid.weights, expected_weights)

    # Unstructured grid
    with pytest.warns(UserWarning):
        grid = CartesianGrid(UnstructuredCoords([[0, 1], [0, 1]], xp=xp))
        assert all_close(grid.weights, xp.asarray(1.0))

def test_grid_serialization():
    grid = CartesianGrid(RegularCoords((1, 1), (10, 10), (0, 0), xp=xp))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = CartesianGrid(SeparatedCoords([xp.arange(10), xp.arange(10, 20)], xp=xp))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = CartesianGrid(UnstructuredCoords([xp.arange(10), xp.arange(10)], xp=xp))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = PolarGrid(UnstructuredCoords([xp.arange(10), xp.arange(10)], xp=xp))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

def test_make_uniform_grid():
    # 1D
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)
    assert grid.shape == (10,)
    assert all_close(grid.delta, 0.1)
    assert all_close(grid.zero, -0.45)

    # 2D
    grid = make_uniform_grid(dims=[10, 20], extent=[1, 2], xp=xp)
    assert grid.shape == (20, 10)
    assert all_close(grid.delta, xp.asarray([0.1, 0.1]))
    assert all_close(grid.zero, xp.asarray([-0.45, -0.95]))

    # 2D with center
    grid = make_uniform_grid(dims=11, extent=1, center=0.5, has_center=True, xp=xp)
    assert all_close((grid.points[0] + grid.points[-1]) / 2, 0.5)

def test_make_pupil_grid():
    grid = make_pupil_grid(dims=10, diameter=1, xp=xp)
    assert grid.shape == (10, 10)
    assert all_close(grid.delta, 0.1)

def test_make_focal_grid_from_pupil_grid():
    pupil_grid = make_pupil_grid(dims=10, diameter=1, xp=xp)
    focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, q=2, num_airy=5)

    assert focal_grid.shape == (20, 20)

def test_make_focal_grid():
    focal_grid = make_focal_grid(q=2, num_airy=5, spatial_resolution=1, xp=xp)

    assert focal_grid.dims == (20, 20)

def test_make_hexagonal_grid():
    grid = make_hexagonal_grid(circum_diameter=1, n_rings=3, xp=xp)

    assert grid.size == 37

def test_make_chebyshev_grid():
    grid = make_chebyshev_grid(dims=[10, 20], xp=xp)

    assert grid.dims == (10, 20)

def test_make_supersampled_grid():
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)
    ss_grid = make_supersampled_grid(grid, oversampling=2)

    assert ss_grid.dims == (20,)
    assert all_close(ss_grid.delta, 0.05)

def test_make_subsampled_grid():
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)
    ss_grid = make_subsampled_grid(grid, undersampling=2)

    assert ss_grid.dims == (5,)
    assert all_close(ss_grid.delta, 0.2)

def test_subsample_field():
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)
    field = grid.ones()

    subsampled_field = subsample_field(field, subsampling=2)

    assert subsampled_field.grid.dims == (5,)
    assert all_close(subsampled_field, 1)

def test_evaluate_supersampled():
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)

    def field_generator(grid):
        return grid.ones()

    field = evaluate_supersampled(field_generator, grid, oversampling=2)
    assert field.grid.dims == (10,)
    assert all_close(field, 1)

def test_make_uniform_vector_field():
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)

    scalar_field = Field(xp.ones(grid.size), grid)
    vector_field = make_uniform_vector_field(scalar_field, jones_vector=[1, 1])

    assert vector_field.tensor_order == 1
    assert vector_field.tensor_shape == (2,)

def test_make_uniform_vector_field_generator():
    def field_generator(grid):
        return grid.ones()

    vector_field_generator = make_uniform_vector_field_generator(field_generator, jones_vector=[1, 1])
    grid = make_uniform_grid(dims=10, extent=1, xp=xp)
    vector_field = vector_field_generator(grid)

    assert vector_field.tensor_order == 1
    assert vector_field.tensor_shape == (2,)
