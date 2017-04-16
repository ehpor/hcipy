from .wavefront_sensor import WavefrontSensor, WavefrontSensorNew
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

def reduce_pyramid_image(img, pupil_mask):
	img = img.shaped

	sub_shape = img.grid.shape // 2

	# Subpupils
	I_a = img[:sub_shape[0], :sub_shape[1]]
	I_b = img[sub_shape[0]:2*sub_shape[0], :sub_shape[1]]
	I_c = img[sub_shape[0]:2*sub_shape[0], sub_shape[1]:2*sub_shape[1]]
	I_d = img[:sub_shape[0], sub_shape[1]:2*sub_shape[1]]

	norm = I_a + I_b + I_c + I_d

	I_1 = (I_a + I_b - I_c - I_d) / norm
	I_2 = (I_a - I_b - I_c + I_d) / norm

	I_1 = I_1.ravel()# * pupil_mask
	I_2 = I_2.ravel()# * pupil_mask

	res = np.column_stack((I_1, I_2))
	res = Field(res, pupil_mask.grid)
	return res

class PyramidWavefrontSensorNew(WavefrontSensorNew):
	def __init__(self, pupil_grid, pupil_mask, pupil_separation, num_pupil_pixels, q, wavelength, refractive_index, detector):
		pupil_diameter = pupil_grid.x[pupil_mask(pupil_grid) != 0].ptp()

		self.detector_grid = make_pupil_grid(2 * pupil_separation * num_pupil_pixels, 2 * pupil_separation * pupil_diameter)
		self.output_grid = make_pupil_grid(pupil_separation * num_pupil_pixels, pupil_separation * pupil_diameter)

		self.pupil_mask = Field(pupil_mask, self.output_grid)

		focal_grid = make_focal_grid(pupil_grid, q=2 * pupil_separation * q, wavelength=wavelength)

		sep = 0.5 * pupil_separation * pupil_diameter
		surf = -sep / (refractive_index(wavelength) - 1) * (np.abs(focal_grid.x) + np.abs(focal_grid.y))
		surf = Field(surf, focal_grid)

		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=wavelength)
		self.pyramid = SurfaceApodizer(surf, refractive_index)
		self.focal_to_pupil = FraunhoferPropagator(focal_grid, self.detector_grid, wavelength_0=wavelength)

		self.detector = detector()

	def integrate(self, wavefront, dt=1, weight=1):
		wf = self.pupil_to_focal(wavefront)
		wf = self.pyramid(wf)
		wf = self.focal_to_pupil(wf)
		self.detector.integrate(wf, dt, weight)

	def read_out(self):
		self.detector_image = self.detector.read_out()

		return reduce_pyramid_image(self.detector_image, self.pupil_mask)