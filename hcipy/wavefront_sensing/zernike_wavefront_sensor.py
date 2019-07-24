from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..optics import PhaseApodizer, Apodizer
from ..aperture import circular_aperture
from ..field import make_pupil_grid, make_focal_grid, Field
import numpy as np

class ZernikeWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.

	Parameters
	----------
	input_grid : Grid
		The grid on which the input wavefront is defined.
	q : scalar
		The focal plane oversampling coefficient.
	wavelength_0 : scalar
		The reference wavelength that determines the physical scales.
	diameter : scalar
		The diameter of the phase dot in terms of lambda/D.
	num_airy : scalar
		The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
	'''
	def __init__(self, input_grid, q=2, wavelength_0=1, diameter=1.06, num_airy=None):

		if not input_grid.is_regular:
			raise ValueError('The input grid must be a regular grid.')

		self.input_grid = input_grid
		D = np.max(input_grid.delta * (input_grid.shape - 1))
		
		if num_airy is None:
			self.num_airy = np.max((input_grid.shape-1)) / 2
		else:
			self.num_airy = num_airy
		
		self.focal_grid = make_focal_grid(q, self.num_airy, reference_wavelength=wavelength_0, pupil_diameter=D, focal_length=1)
		self.wfs_grid = input_grid

		# Make all the optical elements
		self.spatial_filter = Apodizer(circular_aperture(2 * self.num_airy * wavelength_0 / D)(self.focal_grid))
		phase_mask = Field(np.pi/2 * ( self.focal_grid.as_('polar').r <= diameter * wavelength_0 / D), self.focal_grid)
		self.phase_step = PhaseApodizer(phase_mask)

		# Make the propagators
		self.pupil_to_focal = FraunhoferPropagator(input_grid, self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.wfs_grid)

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
		wf = self.phase_step.forward(wf)
		wf = self.focal_to_pupil(wf)

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

		wf = self.focal_to_pupil.backward(wavefront)
		wf = self.phase_step.backward(wf)
		wf = self.focal_to_pupil.backward(wf)

		return wf

class ZernikeWavefrontSensorEstimator(WavefrontSensorEstimator):
	'''Estimates the wavefront slopes from pyramid wavefront sensor images.
	
	Parameters
	----------
	aperture : function
		A function which mask the output phase measurements.
	output_grid : Grid
		The grid on which the output of a pyramid wavefront sensor is sampled.
	reference : Field
		A reference image to subtract from the Zernike wavefront sensor data.
			
	Attributes
	----------
	measurement_grid : Grid
		The grid on which the phase measurements are defined.
	pupil_mask : array_like
		A mask for the phase measurements.
	num_measurements : int
		The number of pixels in the output vector.
	reference : Field
		A reference image to subtract from the Zernike wavefront sensor data.
	'''
	def __init__(self, aperture, output_grid, reference):
		self.measurement_grid = output_grid
		self.pupil_mask = aperture(self.measurement_grid)
		self.reference = reference
		self.num_measurements = int(np.sum(self.pupil_mask > 0))

	def estimate(self, images):
		'''A function which estimates the phase from a Zernike wavefront sensor image.

		Parameters
		----------
		images - List
			A list of scalar intensity fields containing Zernike wavefront sensor images.

		Returns
		-------
		res - Field
			A field with phase estimates.
		'''
		image = images[0]

		intensity_measurements = (image-self.reference).ravel() * self.pupil_mask
		res = Field(intensity_measurements[self.pupil_mask>0], self.pupil_mask.grid)
		return res