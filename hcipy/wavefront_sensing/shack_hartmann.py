from wavefront_sensor import WavefrontSensorOptics
from ..optics import *
from ..propagation import FresnelPropagator

import numpy as np

class ShackHartmannWavefrontSensor(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_array):
        
        # Make propagator
        sh_prop = FresnelPropagator(input_grid, micro_lens_array.focal_length)
        
        # Make optical system
        self.optical_system = OpticalSystem((sh_prop, microlens_array, sh_prop))
        
class SquareShackHartmannWavefrontSensor(ShackHartmannWavefrontSensor):
    ## Helper class to create a Shack-Hartmann WFS with square microlens array
    def __init__(self, input_grid, f_number, N_lenslets):
        
        diameter = 1.0 / N_lenslets
        x = np.arange(-1,1,diameter)
        mla_grid = CartesianGrid(SeparatedCoords((x,x)))
        
        mla_shape = rectangular_aperture(diameter)
        mla = MicroLensArray(input_grid, mla_grid, mla_shape, f_number * diameter)
        
        ShackHartmannWavefrontSensor.__init__(input_grid, mla)