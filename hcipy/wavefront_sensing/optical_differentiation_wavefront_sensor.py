from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..optics import MultiplexedComplexSurfaceApodizer
from ..field import make_pupil_grid, make_focal_grid, Field

import numpy as np


def heaviside_function(x, epsilon=1e-14):
	'''
	The Heaviside function using the half-maximum convention.
	
	Parameters
	----------
	x : array_like
		The values for which the heaviside function needs to be evaluated.
	epsilon : scalar
		The range that is considered zero. The default value is 1e-14.

	Returns
	-------
	h
		The evaluated heaviside function

	'''
	h = (x > 0).astype(np.float)
	h[abs(x)<epsilon] = 0.5
	return h

def create_polarization_odwfs_amplitude_filter(beta):
	'''
	The OD-wfs amplitude filter following [Haffert 2016].
		
	.. [Haffert 2016] S. Y. Haffert, 2016, "Generalised optical differentiation wavefront sensor: 
		a sensitive high dynamic range wavefront sensor," Opt. Express 24, 18986-19007 (2016).

	Parameters
	----------
	beta : scalar
		
	Returns
	-------
	amplitude_filter
		The lambda function that generates the amplitude profile.

	Raises
	------
	ValueError
		If beta is not between 0 and 1.
	'''
	if beta < 0 or beta > 1:
		raise ValueError('Beta should be between 0 and 1.')

	amplitude_filter = lambda x: np.sin( np.pi/2 * (beta * heaviside_function(x) + (1-beta) * (1+x)/2) )
	return amplitude_filter

def create_odwfs_amplitude_filter(beta):
	'''
	The OD-wfs amplitude filter following [Haffert 2016].
		
	.. [Haffert 2016] S. Y. Haffert, 2016, "Generalised optical differentiation wavefront sensor: 
		a sensitive high dynamic range wavefront sensor," Opt. Express 24, 18986-19007 (2016).

	Parameters
	----------
	beta : scalar
		
	Returns
	-------
	amplitude_filter
		The lambda function that generates the amplitude profile.

	Raises
	------
	ValueError
		If beta is not between 0 and 1.
	'''
	if beta < 0 or beta > 1:
		raise ValueError('Beta should be between 0 and 1.')

	amplitude_filter = lambda x: (beta * heaviside_function(x) + (1-beta) * (1+x)/2) / np.sqrt(2)
	return amplitude_filter


def optical_differentiation_surface(filter_size, amplitude_filter, separation, wavelength_0, refractive_index, orientation=0):
	'''A generator function for the complex multiplexed surface of the ODWFS.

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
	orientation : scalar
		The orientation of the amplitude filters. The default value is 0 radian.
	Returns
	----------
	func : function
		The returned function acts on a grid to create the amplitude filters for that grid.

	'''
	def func(grid):
		surface_grid = grid.rotated(orientation)
		# The surfaces which tilt the beam
		surf1 = -separation / (refractive_index(wavelength_0) - 1) * surface_grid.x
		surf2 = -separation / (refractive_index(wavelength_0) - 1) * -surface_grid.x
		surf3 = -separation / (refractive_index(wavelength_0) - 1) * surface_grid.y
		surf4 = -separation / (refractive_index(wavelength_0) - 1) * -surface_grid.y
		
		surf = (Field(surf1, surface_grid), Field(surf2, surface_grid), Field(surf3, surface_grid), Field(surf4, surface_grid))

		# The physical boundaries of the mask
		filter_mask = (np.abs(grid.x) < filter_size) * (np.abs(grid.y) < filter_size)

		# NOTE : be careful with the plus and minus signs of the filters
		# For energy conservation the squared sum of the filters should be <= 1
		# For electric field conservation the second filter has to have opposite sign.
		filter_1 = amplitude_filter(surface_grid.x / filter_size)
		filter_1 *= filter_mask

		filter_2 = -amplitude_filter(-surface_grid.x / filter_size)
		filter_2 *= filter_mask
		
		filter_3 = amplitude_filter(surface_grid.y / filter_size)
		filter_3 *= filter_mask

		filter_4 = -amplitude_filter(-surface_grid.y / filter_size)
		filter_4 *= filter_mask

		amp = (Field(filter_1, surface_grid), Field(filter_2, surface_grid), Field(filter_3, surface_grid), Field(filter_4, surface_grid))

		return MultiplexedComplexSurfaceApodizer(amp, surf, refractive_index)
	return func

class OpticalDifferentiationWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a optical-differentiation wavefront sensor.

	Parameters
	----------
	amplitude_filter : callable
		The function that defines the amplitude filter in the focal plane.
	input_grid : Grid
		The grid on which the input wavefront is defined.
	output_grid : Grid
		The grid on which the output wavefront is defined.
	separation : scalar
		The separation between the pupils. The default takes the input grid extent as separation.
	D : scalar
		The size of the pupil. The default take sthe input grid extent as pupil size.
	wavelength_0 : scalar
		The reference wavelength that determines the physical scales.
	q : scalar
		The focal plane oversampling coefficient. The default uses the minimal required sampling.
	num_airy : scalar
		The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
	refractive_index : callable
		A callable that returns the refractive index as function of wavelength.
		The default is a refractive index of 1.5.
	'''
	def __init__(self, amplitude_filter, input_grid, output_grid, separation=None, D=None, wavelength_0=1, q=None, num_airy=None, refractive_index=lambda x: 1.5):
		if not input_grid.is_regular:
			raise ValueError('The input grid must be a regular grid.')

		self.input_grid = input_grid
		self.output_grid = output_grid

		if D is None:
			D = np.max(input_grid.delta * (input_grid.shape - 1))
		
		if separation is None:
			separation = D

		# Create the intermediate focal grid
		# Oversampling necessary to see all frequencies in the output wavefront sensor plane
		qmin = np.ceil(max(output_grid.x.ptp() / input_grid.x.ptp(), 2))
		if q is None:
			q = qmin 
		elif q < qmin:
			raise ValueError('The requested focal plane sampling is too low to sufficiently sample the wavefront sensor output.')
		
		if num_airy is None:
			self.num_airy = np.max(input_grid.shape - 1) / 2
		else:
			self.num_airy = num_airy

		num_pixels = 2 * int(self.num_airy * q)
		spatial_resolution = wavelength_0 / D
		self.focal_grid = make_pupil_grid(num_pixels, 2 * spatial_resolution * self.num_airy)

		# Make all the optical elements
		self.filter_size = self.num_airy * wavelength_0 / D
		focal_plane_mask = optical_differentiation_surface(self.filter_size, amplitude_filter, separation/np.sqrt(2), wavelength_0, refractive_index, orientation=np.pi/4)
		self.focal_mask = focal_plane_mask(self.focal_grid)

		# Make the propagators
		self.pupil_to_focal = FraunhoferPropagator(self.input_grid, self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid)

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
		wf = self.focal_to_pupil.forward(wf)

		return wf

	
	def backward(self, wavefront):
		'''Propagates a wavefront backwards through the wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf : Wavefront
			The output wavefront.
		'''
		wf = self.pupil_to_focal.backward(wavefront)
		wf = self.focal_mask.backward(wf)
		wf = self.focal_to_pupil.backward(wf)

		return wf

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
