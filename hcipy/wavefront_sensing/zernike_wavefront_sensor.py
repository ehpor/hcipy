from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..optics import PhaseApodizer
from ..aperture import circular_aperture
from ..field import make_uniform_grid, Field

import numpy as np

class ZernikeWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a Zernike wavefront sensor.

	This class uses a propagation scheme similar to that of Lyot coronagraphs
	with a small blocking mask, see [1]_.

	.. [1] Soummer et al. 2007, "Fast computation of Lyot-style
	coronagraph propagation"

	Parameters
	----------
	input_grid : Grid
		The grid on which the input wavefront is defined.
	phase_step : scalar
		The phase of the phase dot of the zernike wavefront sensor. The default is pi/2.
	phase_dot_diameter : scalar
		The diameter of the phase dot. This has units of lambda_0/D.
	num_pix : scalar
		The number of pixels across the phase dot.
	pupil_diameter : scalar
		The diameter of the pupil. This is used for calculating the size of the phase dot.
	reference_wavelength : scalar
		The reference wavelength. This is used for calcualting the size of the phase dot.
	'''
	def __init__(self, input_grid, phase_step=np.pi/2, phase_dot_diameter=1.06, num_pix=128, pupil_diameter=1, reference_wavelength=1):
		self.input_grid = input_grid
		self.output_grid = input_grid

		# Make the phase dot
		phase_dot_diameter *= pupil_diameter / reference_wavelength
		focal_grid = make_uniform_grid([num_pix, num_pix], phase_dot_diameter)
		self.phase_dot = PhaseApodizer(circular_aperture(phase_dot_diameter)(focal_grid))

		# Make the propagator
		self.prop = FraunhoferPropagator(input_grid, focal_grid)

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
		wf_foc = self.prop.forward(wavefront)
		wf_foc.electric_field -= self.phase_dot.forward(wf_foc).electric_field

		pup = self.prop.backward(wf_foc)
		pup.electric_field[:] = wavefront.electric_field - pup.electric_field

		return pup

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
		wf_foc = self.prop.forward(wavefront)
		wf_foc.electric_field -= self.phase_dot.backward(wf_foc).electric_field

		pup = self.prop.backward(wf_foc)
		pup.electric_field[:] = wavefront.electric_field - pup.electric_field

		return pup

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
