import numpy as np
from .apodization import SurfaceApodizer
from .optical_element import OpticalElement
from .surface_profiles import spherical_surface_sag, even_aspheric_surface_sag
from ..field import Field

def closest_points(point_grid, evaluated_grid):
    from scipy import spatial

    tree = spatial.KDTree(point_grid.points)
    d, i = tree.query(evaluated_grid.points)

    return Field(i, evaluated_grid), Field(d, evaluated_grid)

class MicroLensArray(OpticalElement):
    '''A parabolic micro-lens array.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the micro-lens array is evaluated.
    lenslet_grid : Grid
        A grid containing the lenslet positions.
    focal_length : scalar
        The focal length of the micro-lenses
    lenslet_shape : field generator
        The shape of a lenslet.
    '''
    def __init__(self, input_grid, lenslet_grid, focal_length, lenslet_shape=None):
        self.input_grid = input_grid
        self.focal_length = focal_length

        self.mla_grid = lenslet_grid

        if lenslet_shape is None:
            indices, distances = closest_points(lenslet_grid, input_grid)

            self.mla_index = indices
            self.mla_opd = (-1 / (2 * focal_length)) * distances**2
        else:
            self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
            self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)

            for i, (x, y) in enumerate(lenslet_grid.as_('cartesian').points):
                shifted_grid = input_grid.shifted((x, y))
                mask = lenslet_shape(shifted_grid) != 0

                self.mla_opd[mask] = (-1 / (2 * focal_length)) * (shifted_grid.x[mask]**2 + shifted_grid.y[mask]**2)
                self.mla_index[mask] = i

        self.mla_surface = SurfaceApodizer(self.mla_opd, 2)

    def forward(self, wavefront):
        return self.mla_surface.forward(wavefront)

    def backward(self, wavefront):
        return self.mla_surface.backward(wavefront)


class SphericalMicroLensArray(OpticalElement):
    '''An even asphere micro-lens array.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the micro-lens array is evaluated.
    lenslet_grid : Grid
        A grid containing the lenslet positions.
    radius_of_curvature : scalar
        The radius of curvature of the micro-lenses.
    lenslet_shape : field generator
        The shape of a lenslet.
    refractive_index : scalar or function
        The refractive index of the material.
    conic_constant : scalar
        The conic constant of the micro-lenses.
    aspheric_coefficients : array_like
        The aspheric coefficients of the micro-lenses.
    '''
    def __init__(self, input_grid, lenslet_grid, radius_of_curvature, lenslet_shape, refractive_index=1.5):

        self.input_grid = input_grid
        self.mla_grid = lenslet_grid
        self.n = refractive_index
        self.radius_of_curvature = radius_of_curvature

        self.mla_index = -self.input_grid.ones()
        self.mla_opd = self.input_grid.zeros()
        self.surface_sag = spherical_surface_sag(radius_of_curvature)

        for i, (x, y) in enumerate(lenslet_grid.as_('cartesian').points):
            shifted_grid = input_grid.shifted((x, y))
            mask = lenslet_shape(shifted_grid) != 0
            subset_grid = shifted_grid.subset(mask)

            # Check if the micro-lens is within the field of view.
            if np.count_nonzero(mask) > 0:
                self.mla_index[mask] = i
                self.mla_opd[mask] += self.surface_sag(subset_grid)

        self.mla_surface = SurfaceApodizer(self.mla_opd, refractive_index)

    def forward(self, wavefront):
        return self.mla_surface.forward(wavefront)

    def backward(self, wavefront):
        return self.mla_surface.backward(wavefront)

class EvenAsphereMicroLensArray(OpticalElement):
    '''An even asphere micro-lens array.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the micro-lens array is evaluated.
    lenslet_grid : Grid
        A grid containing the lenslet positions.
    radius_of_curvature : scalar
        The radius of curvature of the micro-lenses.
    lenslet_shape : field generator
        The shape of a lenslet.
    refractive_index : scalar or function
        The refractive index of the material.
    conic_constant : scalar
        The conic constant of the micro-lenses.
    aspheric_coefficients : array_like
        The aspheric coefficients of the micro-lenses.
    '''
    def __init__(self, input_grid, lenslet_grid, radius_of_curvature, lenslet_shape, refractive_index=1.5, conic_constant=0, aspheric_coefficients=None):
        self.input_grid = input_grid
        self.mla_grid = lenslet_grid
        self.n = refractive_index
        self.radius_of_curvature = radius_of_curvature
        self.conic_constant = conic_constant
        if aspheric_coefficients is None:
            self.aspheric_coefficients = []
        else:
            self.aspheric_coefficients = aspheric_coefficients

        self.mla_index = -self.input_grid.ones()
        self.mla_opd = self.input_grid.zeros()
        self.surface_sag = even_aspheric_surface_sag(radius_of_curvature, conic_constant, aspheric_coefficients)

        for i, (x, y) in enumerate(lenslet_grid.as_('cartesian').points):
            shifted_grid = input_grid.shifted((x, y))
            mask = lenslet_shape(shifted_grid) != 0
            subset_grid = shifted_grid.subset(mask)

            # Check if the micro-lens is within the field of view.
            if np.count_nonzero(mask) > 0:
                self.mla_index[mask] = i
                self.mla_opd[mask] += self.surface_sag(subset_grid)

        self.mla_surface = SurfaceApodizer(self.mla_opd, refractive_index)

    def forward(self, wavefront):
        return self.mla_surface.forward(wavefront)

    def backward(self, wavefront):
        return self.mla_surface.backward(wavefront)
