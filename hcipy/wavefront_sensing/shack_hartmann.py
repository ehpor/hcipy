from wavefront_sensor import WavefrontSensorOptics
from ..optics import *
from ..propagation import FresnelPropagator

import numpy as np

def calculate_centroids(intensity_field, mla_grid, mla_index):
    pass

class ShackHartmannWavefrontSensor(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_shape, f_number, N_lenslets, wavelength, detector):
        
        diameter = 1.0 / N_lenslets
        x = np.arange(-1,1,diameter)
        self.mla_grid = CartesianGrid(SeparatedCoords((x,x)))
        
        self.wavelength = wavelength
        
        self.detector = detector()
        
        # Make microlens array
        microlens_array = MicroLensArray(input_grid, self.mla_grid, micro_lens_shape, f_number * diameter)
        self.mla_index = microlens_array.mla_index        
        
        # Make propagator
        sh_prop = FresnelPropagator(input_grid, f_number * diameter)
        
        
        self.optical_system = OpticalSystem((microlens_array, sh_prop))
        
    def integrate(self, wavefront, dt=1, weight=1):
        wf = self.optical_system.forward(wavefront)
        self.detector.integrate(wf, dt, weight)
    
    def read_out(self):
        
        # Calculate centroids
        self.detector_image = self.detector.read_out()
        centroids = calculate_centroids(self.detector_image, self.mla_grid, self.mla_index)
        
        # return centroids in np.array()
        return centroids