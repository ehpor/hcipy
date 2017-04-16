from .wavefront_sensor import WavefrontSensor
from ..optics import *
from ..propagation import FraunhoferPropagator
from ..aperture import *
from ..field import *

import numpy as np

class PyramidWavefrontSensor(WavefrontSensor):
	def __init__(self, input_grid, pupil_separation, num_pupil_pixels, detector, measurement_diameter=1, over_sampling=2, wavelength=1):
		
		Din = input_grid.x.max() - input_grid.x.min()
		self.pupil_separation = pupil_separation
		self.num_pupil_pixels = num_pupil_pixels
		self.over_sampling = over_sampling

		self.input_grid = input_grid
		self.output_grid = make_pupil_grid( 2 * pupil_separation * num_pupil_pixels, 2 * pupil_separation * Din)
		self.measurement_grid = make_pupil_grid( pupil_separation * num_pupil_pixels, pupil_separation * Din)
		self.measurement_mask = circular_aperture(Din*measurement_diameter)(self.measurement_grid)>0

		self.detector = detector()

		self.optical_system = self.make_optical_system(self.input_grid, self.output_grid, wavelength)

	def make_optical_system(self, input_grid, output_grid, wavelength):
		fourier_grid = make_focal_grid(input_grid, q=2*self.pupil_separation*self.over_sampling, wavelength=wavelength)

		fraunhofer_1 = FraunhoferPropagator(input_grid, fourier_grid, wavelength_0=wavelength)
		a = 0.5 * self.pupil_separation * 2.0*np.pi / wavelength
		b = 0.5 * self.pupil_separation * 2.0*np.pi / wavelength
		
		T = (fourier_grid.x>0) * (fourier_grid.y>0) * np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y))
		T += (fourier_grid.x>0) * (fourier_grid.y<0) * np.exp(1j* (a * fourier_grid.x - b * fourier_grid.y))
		T += (fourier_grid.x<0) * (fourier_grid.y>0) * np.exp(1j* (-a * fourier_grid.x + b * fourier_grid.y))
		T += (fourier_grid.x<0) * (fourier_grid.y<0) * np.exp(-1j* (a * fourier_grid.x + b * fourier_grid.y))

		horizontal_edge_mask = abs(fourier_grid.x) < 1.0E-13
		vertical_edge_mask = abs(fourier_grid.y) < 1.0E-13
		T[horizontal_edge_mask] += 0.5 * (np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y)) + np.exp(1j* (a * fourier_grid.x - b * fourier_grid.y)))[horizontal_edge_mask]
		T[vertical_edge_mask] += 0.5 * (np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y)) + np.exp(1j* (-a * fourier_grid.x + b * fourier_grid.y)))[vertical_edge_mask]
		T = T/np.abs(T)

		pyramid_prism = Apodizer(T)
		fraunhofer_2 = FraunhoferPropagator(fourier_grid, output_grid, wavelength_0=wavelength)

		return OpticalSystem((fraunhofer_1, pyramid_prism, fraunhofer_2))

	def reduced_pupils(self, pupil_intensity):
		B = np.reshape(pupil_intensity, pupil_intensity.grid.shape)

		Nx = B.shape[0]//2
		Ny = B.shape[1]//2

		Ia = B[0:Nx,0:Ny]
		Ib = B[Nx:(2*Nx),0:Ny]
		Ic = B[Nx:(2*Nx),Ny:(2*Ny)]
		Id = B[0:Nx,Ny:(2*Ny)]

		I1 = (Ia+Ib-Ic-Id)/(Ia+Ib+Ic+Id)
		I2 = (Ia-Ib-Ic+Id)/(Ia+Ib+Ic+Id)
		return I1, I2

	def measurement(self, pupil_intensity):
		I1, I2 = self.reduced_pupils(pupil_intensity)
		return np.concatenate( (I1.flatten()[self.measurement_mask], I2.flatten()[self.measurement_mask]) )