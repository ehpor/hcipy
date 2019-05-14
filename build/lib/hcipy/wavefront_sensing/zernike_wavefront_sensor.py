from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer
from ..field import make_pupil_grid, make_focal_grid, Field
import numpy as np
from matplotlib import pyplot as plt

def phase_step_mask(diameter=1, phase_step=np.pi/2):
	'''Creates a function which can create the phase dot on a grid.

	Parameters
	----------
	diameter : scalar
		The diameter of the phase dot in physical units.
	phase_step : scalar
		The height of the phase dot.
	
	Returns
	----------
	func : function
		The returned function acts on a grid to create the phase mask for that grid.

	'''
	def func(grid):
		radius = grid.as_('polar').r
		phase_mask = Field(phase_step * (radius <= diameter/2), grid)
		return PhaseApodizer(phase_mask)
	return func

class ZernikeWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.

	Parameters
	----------
	pupil_grid : Grid
		The input pupil grid.
	pupil_diameter : scalar
		The input pupil diameter. If this parameter is not set it will take the input grid size as the pupil diameter.
	num_pupil_pixels : int
		The number of pixels that are used to sample the output pupil.
	q : scalar
		The focal plane oversampling coefficient.
	wavelength_0 : scalar
		The reference wavelength which determines the physical scales.
	diameter : scalar
		The diameter of the phase dot in terms of lambda/D.
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
	phase_step : PhaseApodizer
		The filter that is applied in the focal plane.
	'''
	def __init__(self, pupil_grid, pupil_diameter=None, num_pupil_pixels=32, q=2, wavelength_0=1, diameter=1.06, num_airy=None):

		output_grid_size = pupil_grid.x.ptp()
		if pupil_diameter is None:
			pupil_diameter = output_grid_size

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(num_pupil_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.phase_step = phase_step_mask(diameter * wavelength_0/pupil_diameter, np.pi/2)(self.focal_grid)
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
		wf = wavefront.copy()

		wf = self.pupil_to_focal.forward(wf)
		wf = self.phase_step.forward(wf)
		wf = self.focal_to_pupil(wf)

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