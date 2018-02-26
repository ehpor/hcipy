from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer
from ..field import make_pupil_grid, make_focal_grid, Field

import numpy as np

def pyramid_surface(refractive_index, separation, wavelength_0):
	'''Creates a function which can create a pyramid surface on a grid.

	Parameters
	----------
	separation : scalar
		The separation of the pupils in pupil diameters.
	wavelength_0 : scalar
		The reference wavelength for the filter specifications.
	refractive_index : lambda function
		A lambda function for the refractive index which accepts a wavelength.
	
	Returns
	----------
	func : function
		The returned function acts on a grid to create the pyramid surface for that grid.

	'''
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(grid.x) + np.abs(grid.y))
		return SurfaceApodizer(Field(surf, grid), refractive_index)
	return func

class PyramidWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.

	Parameters
	----------
	pupil_grid : Grid
		The input pupil grid.
	pupil_diameter : scalar
		The size of the pupil.
		If it is set to None the pupil_diameter will be the diameter of the pupil_grid.
	pupil_separation : scalar
		The separation distance between the pupils in pupil diameters.
	num_pupil_pixels : int
		The number of pixels that are used to sample the output pupil.
	q : scalar
		The focal plane oversampling coefficient.
	wavelength_0 : scalar
		The reference wavelength which determines the physical scales.
	refractive_index : function
		A function that returns the refractive index as function of wavelength.
	num_airy : int
		The size of the intermediate focal plane grid that is used in terms of lambda/D at the reference wavelength.

	Attributes
	----------
	output_grid : Grid
		The output grid of the wavefront sensor.
	focal_grid : Grid
		The intermediate focal plane grid where the focal plane is sampled.
	pupil_to_focal : FraunhoferPropagator
		A propagator for the input pupil plane to the intermediate focal plane.
	focal_to_pupil : FraunhoferPropagator
		A propagator for the intermediate focal plane to the output pupil plane.
	pyramid : SurfaceApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, pupil_grid, wavelength_0=1, pupil_separation=1.5, pupil_diameter=None, num_pupil_pixels=32, q=4, refractive_index=lambda x : 1.5, num_airy=None):
		if pupil_diameter is None:
			pupil_diameter = pupil_grid.x.ptp()
		
		# Make mask
		sep = 0.5 * pupil_separation * pupil_diameter
		
		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = (pupil_separation + 1) * pupil_diameter
		output_grid_pixels = np.ceil(num_pupil_pixels * (pupil_separation + 1))

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation:
			q = 2 * pupil_separation

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.pyramid = pyramid_surface(refractive_index, sep, wavelength_0)(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wavefront):
		'''Propagates a wavefront through the pyramid wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf : Wavefront
			The output wavefront.
		'''
		wf = self.pupil_to_focal.forward(wf)
		wf = self.pyramid.forward(wf)		
		wf = self.focal_to_pupil(wf)

		return wf

class PyramidWavefrontSensorEstimator(WavefrontSensorEstimator):
	'''Estimates the wavefront slopes from pyramid wavefront sensor images.
	
	Parameters
	----------
	aperture : function
		A function which mask the pupils for the normalized differences.
	output_grid : Grid
		The grid on which the output of a pyramid wavefront sensor is sampled.
			
	Attributes
	----------
	measurement_grid : Grid
		The grid on which the normalized differences are defined.
	pupil_mask : array_like
		A mask for the normalized differences.
	num_measurements : int
		The number of pixels in the output vector.
	'''
	def __init__(self, aperture, output_grid):
		self.measurement_grid = make_pupil_grid(output_grid.shape[0] / 2, output_grid.x.ptp() / 2)
		self.pupil_mask = aperture(self.measurement_grid)
		self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

	def estimate(self, images):
		'''A function which estimates the wavefront slope from a pyramid image.

		Parameters
		----------
		images - List
			A list of scalar intensity fields containing pyramid wavefront sensor images.
			
		Returns
		-------
		res - Field
			A field with wavefront sensor slopes.
		'''
		import warnings
		warnings.warn("This function does not work as expected and will be changed in a future update.", RuntimeWarning)

		image = images.shaped
		sub_shape = image.grid.shape // 2

		# Subpupils
		I_a = image[:sub_shape[0], :sub_shape[1]]
		I_b = image[sub_shape[0]:2*sub_shape[0], :sub_shape[1]]
		I_c = image[sub_shape[0]:2*sub_shape[0], sub_shape[1]:2*sub_shape[1]]
		I_d = image[:sub_shape[0], sub_shape[1]:2*sub_shape[1]]

		norm = I_a + I_b + I_c + I_d

		I_x = (I_a + I_b - I_c - I_d) / norm
		I_y = (I_a - I_b - I_c + I_d) / norm

		I_x = I_x.ravel()[self.pupil_mask>0]
		I_y = I_y.ravel()[self.pupil_mask>0]

		res = Field([I_x, I_y], self.pupil_mask.grid)
		return res