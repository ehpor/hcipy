from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..aperture import circular_aperture
from ..optics import SurfaceApodizer, Apodizer, TipTiltMirror
from ..field import make_pupil_grid, make_focal_grid, Field

import numpy as np

class ModulatedPyramidWavefrontSensor(WavefrontSensorOptics):
	'''The optical elements for a modulated pyramid wavefront sensor.

	Parameters
	----------
	pyramid_wavefront_sensor : WavefrontSensorOptics
		The pyramid wavefront sensor optics that are used.
	modulation : scalar
		The modulation radius in radians.
	num_steps : int
		The number of steps per modulation cycle.
	'''
	def __init__(self, pyramid_wavefront_sensor, modulation, num_steps=12):
		self.modulation = modulation
		self.pyramid_wavefront_sensor = pyramid_wavefront_sensor
		self.tip_tilt_mirror = TipTiltMirror(self.pyramid_wavefront_sensor.input_grid)

		theta = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
		x_modulation = modulation / 2 * np.cos(theta)
		y_modulation = modulation / 2 * np.sin(theta)

		self.modulation_positions = CartesianGrid(UnstructuredCoords(x_modulation, y_modulation))

	def forward(self, wavefront):
		'''Propagates a wavefront through the modulated pyramid wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf_modulated : list
			A list of wavefronts for each modulation position.
		'''
		wf_modulated = []

		for point in self.modulation_positions.points:
			self.tip_tilt_mirror.actuators = point
			modulated_wavefront = self.tip_tilt_mirror.forward(wavefront)

			wf_modulated.append(self.pyramid_wavefront_sensor.forward(modulated_wavefront))

		return wf_modulated

	def backward(self, wavefront):
		raise RuntimeError('This is a non-physical operation.')

class PyramidWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.

	Parameters
	----------
	input_grid : Grid
		The grid on which the input wavefront is defined.
	separation : scalar
		The separation between the pupils. The default takes the input grid extent as separation.
	wavelength_0 : scalar
		The reference wavelength that determines the physical scales.
	q : scalar
		The focal plane oversampling coefficient. The default uses the minimal required sampling.
	refractive_index : callable
		A callable that returns the refractive index as function of wavelength.
		The default is a refractive index of 1.5.
	num_airy : scalar
		The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
	'''
	def __init__(self, input_grid, separation=None, wavelength_0=1, q=None, num_airy=None, refractive_index=lambda x: 1.5):
		if not input_grid.is_regular:
			raise ValueError('The input grid must be a regular grid.')

		self.input_grid = input_grid
		D = np.max(input_grid.delta * (input_grid.shape - 1))
		
		if separation is None:
			separation = D

		# Oversampling necessary to see all frequencies in the output wavefront sensor plane
		qmin = max(2 * separation / D, 1)
		if q is None:
			q = qmin 
		elif q < qmin:
			raise ValueError('The requested focal plane sampling is too low to sufficiently sample the wavefront sensor output.')

		if num_airy is None:
			self.num_airy = np.max(input_grid.shape - 1) / 2
		else:
			self.num_airy = num_airy
		
		self.focal_grid = make_focal_grid(q, self.num_airy, reference_wavelength=wavelength_0, pupil_diameter=D, focal_length=1)
		self.output_grid = make_pupil_grid(qmin * input_grid.dims, qmin * D)
		
		# Make all the optical elements
		self.spatial_filter = Apodizer(circular_aperture(2 * self.num_airy * wavelength_0 / D)(self.focal_grid))
		pyramid_surface = -separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(self.focal_grid.x) + np.abs(self.focal_grid.y))
		self.pyramid = SurfaceApodizer(Field(pyramid_surface, self.focal_grid), refractive_index)

		# Make the propagators
		self.pupil_to_focal = FraunhoferPropagator(input_grid, self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid)

	def forward(self, wavefront):
		'''Propagates a wavefront through the pyramid wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf_wfs : Wavefront
			The output wavefront.
		'''
		wf_focus = self.pupil_to_focal.forward(wavefront)
		wf_pyramid = self.pyramid.forward(self.spatial_filter.forward(wf_focus))
		wf_wfs = self.focal_to_pupil.forward(wf_pyramid)

		return wf_wfs

	def backward(self, wavefront):
		'''Propagates a wavefront backwards through the pyramid wavefront sensor.

		Parameters
		----------		
		wavefront : Wavefront
			The input wavefront that will propagate through the system.

		Returns
		-------
		wf_pupil : Wavefront
			The output wavefront.
		'''
		wf_focus = self.focal_to_pupil.backward(wavefront)
		wf_pyramid = self.pyramid.backward(self.spatial_filter.backward(wf_focus))
		wf_pupil = self.pupil_to_focal.backward(wf_pyramid)

		return wf_pupil

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
	def __init__(self, aperture):
		self.pupil_mask = aperture
		self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

	def estimate(self, images):
		'''A function which estimates the wavefront slope from a pyramid image.

		Parameters
		----------
		images - list
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
