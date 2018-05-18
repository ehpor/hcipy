from ..field import Field
from ..statistics import SpectralNoiseFactoryFFT
from .apodization import SurfaceApodizer

def make_power_law_error(pupil_grid, ptv, diameter, exponent=-2.5, aperture=None, remove_modes=None):
	def psd(grid):
		res = Field(grid.as_('polar').r**-exponent, grid)
		res[grid.as_('polar').r == 0] = 0
		return res
	
	if aperture is None:
		aperture = circular_aperture(diameter)(pupil_grid)
	
	screen = SpectralNoiseFactoryFFT(psd, pupil_grid).make_random()()
	screen *= ptv / np.ptp(screen[aperture != 0])

	if remove_modes is not None:
		modes = make_zernike_basis(remove_modes, diameter, pupil_grid)
		trans = modes.transformation_matrix
		trans_inv = inverse_tikhonov(trans, 1e-6)
		screen -= trans.dot(trans_inv.dot(screen))
	
	return Field(np.exp(2j*np.pi * screen) * aperture, pupil_grid)

class SurfaceError(SurfaceApodizer):
	def __init__(self, pupil_grid, ptv, diameter, exponent=-2.5, refractive_index=-1, aperture=None, remove_modes=None):
		surface = make_power_law_error(pupil_grid, ptv, diameter, exponent, aperture, remove_modes)
		SurfaceApodizer.__init__(self, surface, refractive_index)

class SurfaceAtDistance(OpticalElement):
	def __init__(self, surface_error, distance):
		self.fresnel = FresnelPropagator(pupil_grid, distance)
		self.surface = surface_error

	def forward(self, wavefront):
		wf = self.fresnel.forward(wavefront)
		wf = self.surface.forward(wf)
		return self.fresnel.backward(wf)
	
	def backward(self, wavefront):
		wf = self.fresnel.forward(wavefront)
		wf = self.surface.backward(wf)
		return self.fresnel.backward(wf)
