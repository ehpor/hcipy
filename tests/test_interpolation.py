import numpy as np
import pytest
import hcipy

def make_separated_field():
    grid = hcipy.make_uniform_grid([4, 4], [1, 1])
    values = np.arange(grid.size)

    return hcipy.Field(values, grid)

def make_unstructured_field():
    field = make_separated_field()
    grid = hcipy.CartesianGrid(hcipy.UnstructuredCoords((field.grid.x, field.grid.y)))

    return hcipy.Field(field, grid)

def make_separated_grid():
    return hcipy.make_uniform_grid([2, 2], [0.8, 0.8])

def make_unstructured_grid():
    grid = make_separated_grid()
    return hcipy.CartesianGrid(hcipy.UnstructuredCoords((grid.x, grid.y)))

fields = [
    pytest.param(make_separated_field(), id='separated_in'),
    pytest.param(make_unstructured_field(), id='unstructured_in'),
]

grids = [
    pytest.param(make_separated_grid(), id='separated_out'),
    pytest.param(make_unstructured_grid(), id='unstructured_out'),
]

@pytest.mark.parametrize('grid_out', grids)
@pytest.mark.parametrize('field_in', fields)
def test_linear_interpolator(field_in, grid_out):
    interpolator = hcipy.make_linear_interpolator(field_in)
    field_out = interpolator(grid_out)

    expected_values = np.array([3.5, 5.1, 9.9, 11.5])
    assert np.allclose(field_out, expected_values)

@pytest.mark.parametrize('grid_out', grids)
@pytest.mark.parametrize('field_in', fields)
def test_nearest_interpolator(field_in, grid_out):
    interpolator = hcipy.make_nearest_interpolator(field_in)
    field_out = interpolator(grid_out)

    expected_values = np.array([5, 6, 9, 10])
    assert np.allclose(field_out, expected_values)
