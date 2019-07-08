import numpy as np

from ..field import Field
from ..statistics import SpectralNoiseFactoryFFT
from .apodization import SurfaceApodizer
from ..propagation import FresnelPropagator
from ..mode_basis import make_zernike_basis
from ..math_util import inverse_tikhonov
from ..aperture import circular_aperture
from .optical_element import OpticalElement

def make_power_law_error(pupil_grid, ptv, diameter, exponent=-2.5, aperture=None, remove_modes=None):
	'''Create an error surface from a power-law power spectral density.

	Parameters
	----------
	pupil_grid : Grid
		The grid on which to calculate the error.
	ptv : scalar
		The peak-to-valley of the wavefront aberration in meters.
	diameter : scalar
		The diameter over which the ptv is calculated.
	exponent : scalar
		The exponent of the power law.
	aperture : Field
		The mask over which to calculate the ptv. A circular aperture with diameter
		`diameter` is used if this is not given.
	remove_modes : ModeBasis
		The modes which to remove from the surface aberration. The peak-to-valley
		is enforced before these modes are removed. This allows for correctting surface
		errors with optic alignment.
	
	Returns
	-------
	Field
		The surface error calculated on `pupil_grid`.
	'''
	def psd(grid):
		res = Field(grid.as_('polar').r**exponent, grid)
		res[grid.as_('polar').r == 0] = 0
		return res
	
	if aperture is None:
		aperture = circular_aperture(diameter)(pupil_grid)
	
	screen = SpectralNoiseFactoryFFT(psd, pupil_grid).make_random()()
	screen *= ptv / np.ptp(screen[aperture != 0])

	if remove_modes is not None:
		trans = remove_modes.transformation_matrix
		trans_inv = inverse_tikhonov(trans, 1e-6)
		screen -= trans.dot(trans_inv.dot(screen))
	
	return Field(screen * aperture, pupil_grid)

class SurfaceAberration(SurfaceApodizer):
	'''A surface aberration with a specific power law.

	Parameters
	----------
	pupil_grid : Grid
		The grid on which the incoming wavefront is defined.
	ptv : scalar
		The peak-to-valley of the wavefront aberration in meters.
	diameter : scalar
		The diameter over which the ptv is calculated.
	exponent : scalar
		The exponent of the power law.
	refractive_index : scalar
		The refractive index of the surface for which this is the surface error.
		The default is a mirror surface.
	aperture : Field
		The mask over which to calculate the ptv. A circular aperture with diameter
		`diameter` is used if this is not given.
	remove_modes : ModeBasis
		The modes which to remove from the surface aberration. The peak-to-valley
		is enforced before these modes are removed. This allows for correctting surface
		errors with optic alignment.
	'''
	def __init__(self, pupil_grid, ptv, diameter, exponent=-2.5, refractive_index=-1, aperture=None, remove_modes=None):
		surface = make_power_law_error(pupil_grid, ptv, diameter, exponent, aperture, remove_modes)
		SurfaceApodizer.__init__(self, surface, refractive_index)

class SurfaceAberrationAtDistance(OpticalElement):
	'''A surface at a certain distance from the current plane.

	Light is propagated to this surface, then the surface errors are
	applied, and afterwards the light is propagated back towards the
	original plane. This allows for easy addition of surface errors on
	lenses, while still retaining the Fraunhofer propagations in between
	focal and pupil planes.

	Parameters
	----------
	surface_aberration : OpticalElement
		The optical element describing the surface aberration.
	distance : scalar
		The distance from the current plane.
	'''
	def __init__(self, surface_aberration, distance):
		self.fresnel = FresnelPropagator(surface_aberration.input_grid, distance)
		self.surface_aberration = surface_aberration

	def forward(self, wavefront):
		'''Propagate a wavefront forwards through the surface aberration.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront after the surface aberration. This wavefront is
			given at the same plane as `wavefront`.
		'''
		wf = self.fresnel.forward(wavefront)
		wf = self.surface_aberration.forward(wf)
		return self.fresnel.backward(wf)
	
	def backward(self, wavefront):
		'''Propagate a wavefront backwards through the surface aberration.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront before the surface aberration. This wavefront is
			given at the same plane as `wavefront`.
		'''
		wf = self.fresnel.forward(wavefront)
		wf = self.surface_aberration.backward(wf)
		return self.fresnel.backward(wf)
