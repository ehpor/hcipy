from wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import *
from ..field import *
from ..aperture import *
from ..propagation import FresnelPropagator

import numpy as np

class ShackHartmannWavefrontSensorOptics(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_array):
        
        # Make propagator
        sh_prop = FresnelPropagator(input_grid, micro_lens_array.focal_length)
        
        # Make optical system
        OpticalSystem.__init__(self, (sh_prop, micro_lens_array, sh_prop))
        self.mla_index = micro_lens_array.mla_index
        
class SquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):
    ## Helper class to create a Shack-Hartmann WFS with square microlens array
    def __init__(self, input_grid, f_number, N_lenslets):
        
        diameter = 1.0 / N_lenslets
        x = np.arange(-1,1,diameter)
        mla_grid = CartesianGrid(SeparatedCoords((x,x)))
        
        mla_shape = rectangular_aperture(diameter)
        mla = MicroLensArray(input_grid, mla_grid, mla_shape, f_number * diameter)
        
        ShackHartmannWavefrontSensorOptics.__init__(self, input_grid, mla)
        
  
class ShackHartmannWavefrontSensorEstimator(WavefrontSensorEstimator):
    def __init__(self, mla_index):
        self.mla_index = mla_index
        
    def estimate(self, images):
        image = images[0]
        centroids = np.empty([2 * np.unique(self.mla_index).size, 1])
        cent_index = 0
        
        for sub_index in np.unique(self.mla_index):
            
            # Select individual subapertures based on mla_index
            subaperture = image[self.mla_index == sub_index]
            
            # Find the centroid of the subaperture
            x = image.grid.x[self.mla_index == sub_index]
            y = image.grid.y[self.mla_index == sub_index]
            centx = sum(subaperture * x) / sum(subaperture)
            centy = sum(subaperture * y) / sum(subaperture)
            
            # Add the centroid coordinates to the numpy array            
            centroids[cent_index] = centx
            cent_index += 1
            centroids[cent_index] = centy
            cent_index += 1
            