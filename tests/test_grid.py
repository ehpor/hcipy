from hcipy import *
from hcipy.field.coordinates import Coords
import numpy as np
import pytest

def test_unstructured_coords():
    coords = UnstructuredCoords([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert coords.size == 3
    assert len(coords) == 2
    assert np.all(coords[0] == [1, 2, 3])
    assert np.all(coords[1] == [4, 5, 6])

    coords_copy = coords.copy()
    assert coords == coords_copy
    coords_copy += 1
    assert coords != coords_copy
    assert np.allclose(coords_copy[0], [2, 3, 4])
    assert np.allclose(coords_copy[1], [5, 6, 7])

    coords_copy = coords.copy()
    coords_copy *= 2
    assert np.allclose(coords_copy[0], [2, 4, 6])
    assert np.allclose(coords_copy[1], [8, 10, 12])

    coords.reverse()
    assert np.all(coords[0] == [3, 2, 1])
    assert np.all(coords[1] == [6, 5, 4])

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_separated_coords():
    coords = SeparatedCoords([[1.0, 2.0, 3.0], [4.0, 5.0]])
    assert coords.size == 6
    assert len(coords) == 2
    assert np.all(coords.dims == [3, 2])
    assert np.all(coords.shape == (2, 3))
    assert np.all(coords.separated_coords[0] == [1, 2, 3])
    assert np.all(coords.separated_coords[1] == [4, 5])
    assert np.allclose(coords[0], [1, 2, 3, 1, 2, 3])
    assert np.allclose(coords[1], [4, 4, 4, 5, 5, 5])

    coords_copy = coords.copy()
    assert coords == coords_copy

    coords_copy += np.array([1, 2])
    assert coords != coords_copy
    assert np.allclose(coords_copy.separated_coords[0], [2, 3, 4])
    assert np.allclose(coords_copy.separated_coords[1], [6, 7])

    coords_copy = coords.copy()
    coords_copy *= 2
    assert np.allclose(coords_copy.separated_coords[0], [2, 4, 6])
    assert np.allclose(coords_copy.separated_coords[1], [8, 10])

    coords.reverse()
    assert np.all(coords.separated_coords[0] == [3, 2, 1])
    assert np.all(coords.separated_coords[1] == [5, 4])

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_regular_coords():
    coords = RegularCoords(delta=[0.5, 1], dims=[3, 2], zero=[0, 0])
    assert coords.size == 6
    assert len(coords) == 2
    assert np.all(coords.dims == [3, 2])
    assert np.all(coords.shape == (2, 3))
    assert np.all(coords.delta == [0.5, 1])
    assert np.all(coords.zero == [0, 0])

    assert np.allclose(coords.separated_coords[0], [0, 0.5, 1])
    assert np.allclose(coords.separated_coords[1], [0, 1])

    assert np.allclose(coords[0], [0, 0.5, 1, 0, 0.5, 1])
    assert np.allclose(coords[1], [0, 0, 0, 1, 1, 1])

    coords_copy = coords.copy()
    assert coords == coords_copy
    coords_copy += np.array([1, 2])
    assert coords != coords_copy
    assert np.allclose(coords_copy.zero, [1, 2])

    coords_copy = coords.copy()
    coords_copy *= 2
    assert np.allclose(coords_copy.delta, [1, 2])
    assert np.allclose(coords_copy.zero, [0, 0])

    coords.reverse()
    assert np.allclose(coords.delta, [-0.5, -1])
    assert np.allclose(coords.zero, [1, 1])

    d = coords.to_dict()
    coords2 = Coords.from_dict(d)
    assert coords == coords2

def test_coords_arithmetic():
    # Unstructured
    c1 = UnstructuredCoords([[1.0, 2.0], [3.0, 4.0]])
    c2 = c1 + 1
    assert np.allclose(c2[0], [2, 3]) and np.allclose(c2[1], [4, 5])
    c2 = c1 - 1
    assert np.allclose(c2[0], [0, 1]) and np.allclose(c2[1], [2, 3])
    c2 = c1 * 2
    assert np.allclose(c2[0], [2, 4]) and np.allclose(c2[1], [6, 8])
    c2 = c1 / 2
    assert np.allclose(c2[0], [0.5, 1]) and np.allclose(c2[1], [1.5, 2])

    # Separated
    c1 = SeparatedCoords([[1.0, 2.0], [3.0, 4.0]])
    c2 = c1 + np.array([1, 2])
    assert np.allclose(c2.separated_coords[0], [2, 3]) and np.allclose(c2.separated_coords[1], [5, 6])
    c2 = c1 - np.array([1, 2])
    assert np.allclose(c2.separated_coords[0], [0, 1]) and np.allclose(c2.separated_coords[1], [1, 2])
    c2 = c1 * 2
    assert np.allclose(c2.separated_coords[0], [2, 4]) and np.allclose(c2.separated_coords[1], [6, 8])
    c2 = c1 / 2
    assert np.allclose(c2.separated_coords[0], [0.5, 1]) and np.allclose(c2.separated_coords[1], [1.5, 2])

    # Regular
    c1 = RegularCoords([1, 2], [3, 4], [5, 6])
    c2 = c1 + np.array([1, 2])
    assert np.allclose(c2.zero, [6, 8])
    c2 = c1 - np.array([1, 2])
    assert np.allclose(c2.zero, [4, 4])
    c2 = c1 * 2
    assert np.allclose(c2.delta, [2, 4]) and np.allclose(c2.zero, [10, 12])
    c2 = c1 / 2
    assert np.allclose(c2.delta, [0.5, 1]) and np.allclose(c2.zero, [2.5, 3])

def test_grid_creation():
    # Unstructured
    ug = Grid(UnstructuredCoords([[1, 2], [3, 4]]))
    assert ug.is_unstructured
    assert not ug.is_separated
    assert not ug.is_regular
    assert ug.ndim == 2
    assert ug.size == 2

    # Separated
    sg = Grid(SeparatedCoords([[1, 2], [3, 4]]))
    assert not sg.is_unstructured
    assert sg.is_separated
    assert not sg.is_regular
    assert sg.ndim == 2
    assert sg.size == 4
    assert np.all(sg.dims == [2, 2])
    assert np.all(sg.shape == (2, 2))

    # Regular
    rg = Grid(RegularCoords([1, 1], [2, 2]))
    assert not rg.is_unstructured
    assert rg.is_separated
    assert rg.is_regular
    assert rg.ndim == 2
    assert rg.size == 4
    assert np.all(rg.dims == [2, 2])
    assert np.all(rg.shape == (2, 2))
    assert np.all(rg.delta == [1, 1])
    assert np.all(rg.zero == [0, 0])

def test_grid_exceptions():
    ug = Grid(UnstructuredCoords([[1, 2], [3, 4]]))
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

    sg = Grid(SeparatedCoords([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        sg.delta
    with pytest.raises(ValueError):
        sg.zero
    with pytest.raises(ValueError):
        sg.regular_coords

def test_grid_subset():
    grid = CartesianGrid(RegularCoords([1, 1], [10, 10]))
    subset_grid = grid.subset(grid.x > 3)
    assert subset_grid.size < grid.size
    assert np.all(subset_grid.x > 3)

    indices = np.where(grid.x > 3)
    subset_grid2 = grid.subset(indices)
    assert subset_grid == subset_grid2

def test_cartesian_grid_transformations():
    grid = CartesianGrid(RegularCoords([1.0], 8))

    # Scaling
    sgrid = grid.scaled(2)
    assert np.allclose(sgrid.delta, 2)
    sgrid.scale(0.5)
    assert np.allclose(sgrid.delta, 1)
    assert grid == sgrid

    # Shifting
    sgrid = grid.shifted(2)
    assert np.allclose(sgrid.zero, 2)
    sgrid.shift(-2)
    assert np.allclose(sgrid.zero, 0)
    assert grid == sgrid

    # Rotation 2D
    grid2d = CartesianGrid(RegularCoords([1, 1], [3, 3], zero=[-1, -1]))
    rgrid = grid2d.rotated(np.pi / 2)
    assert np.allclose(rgrid.points, np.array([[1, 1, 1, 0, 0, 0, -1, -1, -1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]]).T)

    # Rotation 3D (around the z-axis in this test)
    grid3d = CartesianGrid(RegularCoords([1, 1, 1], [2, 2, 2]))
    rgrid = grid3d.rotated(np.pi / 2, axis=[0, 0, 1])
    assert np.allclose(rgrid.x, -grid3d.y)
    assert np.allclose(rgrid.y, grid3d.x)
    assert np.allclose(rgrid.z, grid3d.z)

def test_polar_grid_transformations():
    grid = PolarGrid(UnstructuredCoords([[1.0, 2.0], [np.pi / 4, np.pi / 2]]))

    # Scaling
    sgrid = grid.scaled(2)
    assert np.allclose(sgrid.r, [2, 4])
    sgrid.scale(0.5)
    assert np.allclose(sgrid.r, [1, 2])

    # Rotation
    rgrid = grid.rotated(np.pi / 4)
    assert np.allclose(rgrid.theta, [np.pi / 2, 3 * np.pi / 4])
    rgrid.rotate(-np.pi / 4)
    assert np.allclose(rgrid.theta, [np.pi / 4, np.pi / 2])

    # Shifting
    sgrid = grid.shifted([1, 0])
    assert sgrid.is_('cartesian')
    # Check if the point (r=1, th=pi/4) which is (x=sqrt(2)/2, y=sqrt(2)/2) is shifted
    # to (x=1+sqrt(2)/2, y=sqrt(2)/2)
    assert np.allclose(sgrid.x[0], 1 + np.sqrt(2) / 2)
    assert np.allclose(sgrid.y[0], np.sqrt(2) / 2)

def test_coordinate_transformation():
    # 2D
    grid_cart = CartesianGrid(RegularCoords([1, 1], [3, 3]))
    grid_pol = grid_cart.as_('polar')
    assert grid_pol.is_('polar')
    grid_cart2 = grid_pol.as_('cartesian')
    assert grid_cart2.is_('cartesian')
    assert np.allclose(grid_cart.points, grid_cart2.points)

def test_grid_reversal():
    grid = CartesianGrid(RegularCoords([1, 1], [10, 12]))
    rev_grid = grid.reversed()
    assert not grid == rev_grid
    assert np.allclose(grid.points, rev_grid.points[::-1])
    rev_grid.reverse()
    assert grid == rev_grid

def test_grid_closest_to():
    grid = CartesianGrid(RegularCoords([1, 1], [10, 10]))
    p = [2.1, 3.8]
    idx = grid.closest_to(p)
    assert idx == 42  # Corresponds to point (2, 4).

def test_grid_field_creation():
    grid = CartesianGrid(RegularCoords([1], 10))
    f_zeros = grid.zeros()
    f_ones = grid.ones()
    f_empty = grid.empty()

    assert np.all(f_zeros == 0)
    assert np.all(f_ones == 1)

    assert f_zeros.grid == grid
    assert f_ones.grid == grid
    assert f_empty.grid == grid

    assert np.all(f_zeros.shape == (10,))
    assert np.all(f_ones.shape == (10,))
    assert np.all(f_empty.shape == (10,))

    f_zeros_t = grid.zeros(tensor_shape=(2, 3))

    assert np.all(f_zeros_t.tensor_shape == (2, 3))
    assert f_zeros_t.shape == (2, 3, 10)

def test_grid_weights():
    # Regular grid
    grid = CartesianGrid(RegularCoords([0.1, 0.2], [10, 20]))
    assert np.allclose(grid.weights, 0.1 * 0.2)

    # Separated grid
    x = np.array([-1, 0, 1, 3])
    y = np.array([10, 20])
    grid = CartesianGrid(SeparatedCoords([x, y]))

    w_x = np.concatenate(([x[1] - x[0]], (x[2:] - x[:-2]) / 2, [x[-1] - x[-2]]))
    w_y = np.concatenate(([y[1] - y[0]], (y[2:] - y[:-2]) / 2, [y[-1] - y[-2]]))

    expected_weights = np.outer(w_y, w_x).ravel()
    assert np.allclose(grid.weights, expected_weights)

    # Unstructured grid
    with pytest.warns(UserWarning):
        grid = CartesianGrid(UnstructuredCoords([[0, 1], [0, 1]]))
        assert np.allclose(grid.weights, 1)

def test_grid_serialization():
    grid = CartesianGrid(RegularCoords([1, 1], [10, 10], [0, 0]))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = CartesianGrid(SeparatedCoords([np.arange(10), np.arange(10, 20)]))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = CartesianGrid(UnstructuredCoords([np.random.rand(10), np.random.rand(10)]))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2

    grid = PolarGrid(UnstructuredCoords([np.random.rand(10), np.random.rand(10)]))
    d = grid.to_dict()
    grid2 = Grid.from_dict(d)
    assert grid == grid2
