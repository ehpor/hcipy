from wavefront_sensor import WavefrontSensorOptics
from ..optics import *
from ..propagation import FresnelPropagator

import numpy as np

def calculate_centroids(intensity_field, mla_grid, mla_index):
    pass

class ShackHartmannWavefrontSensor(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_shape, f_number, diameter, N_lenslets, wavelength, detector):
        self.input_grid = input_grid
        x = np.arange(-1,1,diameter)
        self.mla_grid = CartesianGrid(SeparatedCoords((x,x)))
        self.mla_shape = micro_lens_shape
        self.F_mla = f_number
        self.D_mla = diameter
        self.N_lenslets = N_lenslets
        self.wavelength = wavelength
        
        self.detector = detector()        
        
        self.optical_system = self.make_optical_system()
        
    
    def make_optical_system(self):
        
        # Make microlens array
        microlens_array = MicroLensArray(self.input_grid, self.mla_grid, self.mla_shape, self.F * self.D)
        self.mla_index = microlens_array.mla_index        
        
        # Make propagator
        sh_prop = FresnelPropagator(self.input_grid, self.F_mla * self.D_mla)
        
        # Make optical system        
        return OpticalSystem((microlens_array, sh_prop))
        
    def integrate(self, wavefront, dt=1, weight=1):
        wf = self.optical_system.forward(wavefront)
        self.detector.integrate(wf, dt, weight)
    
    def read_out(self):
        # Calculate centroids
        self.detector_image = self.detector.read_out()
        centroids = calculate_centroids(self.detector_image, self.mla_grid, self.mla_index)
        # return centroids in np.array()
        return centroids