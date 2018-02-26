from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer, MultiplexedComplexSurfaceApodizer
from ..field import make_pupil_grid, make_focal_grid, Field

import numpy as np

def optical_differentiation_surface(filter_size, amplitude_filter, separation, wavelength_0, refractive_index):
	'''Creates a function which can create the complex multiplexed surface for the ODWFS on a grid.

	Parameters
	----------
	filter_size : scalar
		The physical size of the filter in lambda/D.
	amplitude_filter : lambda function
		A lambda function which acts on the focal plane grid to create a amplitude filter.
	separation : scalar
		The separation of the pupils in pupil diameters.
	wavelength_0 : scalar
		The reference wavelength for the filter specifications.
	refractive_index : lambda function
		A lambda function for the refractive index which accepts a wavelength.
	
	Returns
	----------
	func : function
		The returned function acts on a grid to create the amplitude filters for that grid.

	'''
	def func(grid):
		# The surfaces which tilt the beam
		# This positions the pupils
		surf1 = -separation / (refractive_index(wavelength_0) - 1) * grid.x
		surf2 = -separation / (refractive_index(wavelength_0) - 1) * -grid.x
		surf3 = -separation / (refractive_index(wavelength_0) - 1) * grid.y
		surf4 = -separation / (refractive_index(wavelength_0) - 1) * -grid.y
		
		surf = (Field(surf1, grid), Field(surf2, grid), Field(surf3, grid), Field(surf4, grid))

		# The physical boundaries of the mask
		filter_mask = (np.abs(grid.x) < filter_size) * (np.abs(grid.y) < filter_size)

		x_mask = np.abs(grid.x) < 1e-15
		y_mask = np.abs(grid.y) < 1e-15

		# NOTE : be carefull with the plus and minus signs of the filters
		# For energy conservation the squared sum of the filters should be <= 1
		# For electric field conservation the second filter has to have opposite sign.
		filter_1 = amplitude_filter(grid.x / filter_size)
		filter_1[x_mask] = 1 / 2
		filter_1 *= filter_mask

		filter_2 = -amplitude_filter(-grid.x / filter_size)
		filter_2[x_mask] = -1 / 2
		filter_2 *= filter_mask
		
		filter_3 = amplitude_filter(grid.y / filter_size)
		filter_3[y_mask] = 1 / 2
		filter_3 *= filter_mask

		filter_4 = -amplitude_filter(-grid.y / filter_size)
		filter_4[y_mask] = -1 / 2
		filter_4 *= filter_mask

		amp = (Field(filter_1, grid), Field(filter_2, grid), Field(filter_3, grid), Field(filter_4, grid))

		return MultiplexedComplexSurfaceApodizer(amp, surf, refractive_index)
	return func

class OpticalDifferentiationWavefrontSensorOptics(WavefrontSensorOptics):
	'''The value of some physical quantity for each point in some coordinate system.

	Parameters
	----------
	filter_size : scalar
		The physical size of the filter in lambda/D.
	amplitude_filter : lambda function
		A lambda function which acts on the focal plane grid to create a amplitude filter.
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
	focal_mask : MultiplexedComplexSurfaceApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, filter_size, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy=None):
		# Make mask
		if pupil_diameter is None:
			pupil_diameter = pupil_grid.x.ptp()

		# Multiply by 2 because we want to have two pupils next to each other
		sep = 0.5 * pupil_separation * pupil_diameter * np.sqrt(2)
		output_grid_size = (pupil_separation * np.sqrt(2) + 1) * pupil_diameter
		output_grid_pixels = np.ceil(num_pupil_pixels * (pupil_separation + 1) * np.sqrt(2))

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation * np.sqrt(2):
			q = 2 * pupil_separation * np.sqrt(2)
		
		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		if filter_size is None:
			filter_size = self.focal_grid.x.max()
		else:
			filter_size = filter_size * wavelength_0 / pupil_diameter

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		focal_plane_mask = optical_differentiation_surface(filter_size, amplitude_filter, sep, wavelength_0, refractive_index)
		self.focal_mask = focal_plane_mask(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wavefront):
		'''Propagates a wavefront through the wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf : Wavefront
			The output wavefront.
		'''
		wf = self.pupil_to_focal.forward(wavefront)
		wf = self.focal_mask.forward(wf)		
		wf = self.focal_to_pupil(wf)

		return wf

class RooftopWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	'''A rooftop wavefront sensor.

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
	focal_mask : MultiplexedComplexSurfaceApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, pupil_grid, pupil_diameter=None, pupil_separation=1.5, num_pupil_pixels=32, q=4, wavelength_0=1, refractive_index=lambda x: 1.5, num_airy=None):
		amplitude_filter = lambda x: (x < 0) / np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, num_airy, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)

class gODWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	'''A generalised optical differentiation wavefront sensor based on linear amplitude filters.

	Parameters
	----------
	filter_size : scalar
		The physical size of the filter in lambda/D.
	beta : scalar
		The step size. A value of 0 corresponds to a linear filter and a value of 1 to a
		full step.
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
	focal_mask : MultiplexedComplexSurfaceApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, filter_size, beta, pupil_grid, pupil_diameter=None, pupil_separation=1.5, num_pupil_pixels=32, q=4, wavelength_0=1, refractive_index=lambda x: 1.5, num_airy=None):
		amplitude_filter = lambda x: ((x > 0) * beta + (1 - beta) * (1 + x) / 2) / np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, filter_size, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)

class PolgODWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	'''A generalised optical differentiation wavefront sensor based on linear amplitude filters.

	Parameters
	----------
	filter_size : scalar
		The physical size of the filter in lambda/D.
	beta : scalar
		The step size. A value of 0 corresponds to a linear filter and a value of 1 to a
		full step.
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
	focal_mask : MultiplexedComplexSurfaceApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, filter_size, beta, pupil_grid, pupil_diameter=None, pupil_separation=1.5, num_pupil_pixels=32, q=4, wavelength_0=1, refractive_index=lambda x: 1.5, num_airy=None):
		amplitude_filter = lambda x: np.sin(np.pi/2 * ((x > 0) * beta + (1 - beta) * (1 + x) / 2)) / np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, filter_size, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)

class OpticalDifferentiationWavefrontSensorEstimator(WavefrontSensorEstimator):
	'''Estimates the wavefront slopes from OD wavefront sensor images.
	
	Parameters
	----------
	aperture : function
		A function which mask the pupils for the normalized differences.
	output_grid : Grid
		The grid on which the output of an OD wavefront sensor is sampled.
			
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
		'''A function which estimates the wavefront slope from a ODWFS image.

		Parameters
		----------
		images : List
			A list of scalar intensity fields containing OD wavefront sensor images.
			
		Returns
		-------
		res : Field
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

		I_x = (I_d - I_a) / (I_a + I_d)
		I_y = (I_c - I_b) / (I_c + I_b)
		
		I_x = I_x.ravel()[self.pupil_mask>0]
		I_y = I_y.ravel()[self.pupil_mask>0]

		res = Field([I_x, I_y], self.pupil_mask.grid)
		return res