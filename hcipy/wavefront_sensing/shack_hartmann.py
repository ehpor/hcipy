from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import OpticalSystem, MicroLensArray
from ..field import CartesianGrid, Field, SeparatedCoords
from ..propagation import FresnelPropagator

import numpy as np
from scipy import ndimage

class ShackHartmannWavefrontSensorOptics(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_array):
        # Make propagator
        sh_prop = FresnelPropagator(input_grid, micro_lens_array.focal_length)

        # Make optical system
        OpticalSystem.__init__(self, (micro_lens_array, sh_prop))
        self.mla_index = micro_lens_array.mla_index
        self.mla_grid = micro_lens_array.mla_grid
        self.micro_lens_array = micro_lens_array

class SquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):
    ## Helper class to create a Shack-Hartmann WFS with square microlens array
    def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter, offset=None):
        '''
        Create a square Shack-Hartmann wavefront sensor optics.

        MLA offset control is achieved as:
        - Default: centering (offset=None)
            - if num_lenslets is even, a lenslet corner is at (0,0) coordinates.
            - if num_lenslets is odd, a lenslet center is at (0,0) coordinates.
            - equivalent of offset = (0.5, 0.5) if num_lenslet is even else (0,0)
        - Legacy behavior: set offset=(0,0). A lenslet center is at (0,0) regardless of parity
        - Offset parameter: specify (x,y) offset of the MLA grid in lenslet pitch units.
            This is specified by setting a lenslet center at (x,y) coordinates.
        '''
        lenslet_diameter = float(pupil_diameter) / num_lenslets

        if offset is None:
            _o = lenslet_diameter / 2 if num_lenslets % 2 == 0 else 0
            offset_coords = (_o, _o)
        else: # assert offset: tuple[int, int]
            x,y = offset
            offset_coords = (x * lenslet_diameter, y * lenslet_diameter)

        x = np.arange(-pupil_diameter + offset_coords[0],
                      +pupil_diameter, lenslet_diameter)
        y = np.arange(-pupil_diameter + offset_coords[1],
                      +pupil_diameter, lenslet_diameter)
        self.mla_grid = CartesianGrid(SeparatedCoords((x, y)))

        focal_length = f_number * lenslet_diameter
        self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, focal_length)

        ShackHartmannWavefrontSensorOptics.__init__(self, input_grid, self.micro_lens_array)

class ShackHartmannWavefrontSensorEstimator(WavefrontSensorEstimator):
    def __init__(self, mla_grid, mla_index, estimation_subapertures=None):
        self.mla_grid = mla_grid
        self.mla_index = mla_index
        if estimation_subapertures is None:
            self.estimation_subapertures = np.unique(self.mla_index)
        else:
            self.estimation_subapertures = np.flatnonzero(np.array(estimation_subapertures))
        self.estimation_grid = self.mla_grid.subset(estimation_subapertures)

    def estimate(self, images):
        image = images[0]

        fluxes = ndimage.measurements.sum(image, self.mla_index, self.estimation_subapertures)
        sum_x = ndimage.measurements.sum(image * image.grid.x, self.mla_index, self.estimation_subapertures)
        sum_y = ndimage.measurements.sum(image * image.grid.y, self.mla_index, self.estimation_subapertures)

        centroid_x = sum_x / fluxes
        centroid_y = sum_y / fluxes

        centroids = np.array((centroid_x, centroid_y)) - np.array(self.mla_grid.points[self.estimation_subapertures, :]).T
        return Field(centroids, self.estimation_grid)
