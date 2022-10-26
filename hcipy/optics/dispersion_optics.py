from .apodization import SurfaceApodizer, PhaseApodizer
import numpy as np

class ThinPrism(SurfaceApodizer):
	'''A thin prism that operates in the paraxial regime.

	Parameters
	----------
	wedge_angle : scalar
		The wedge angle of the prism.
	refractive_index : scalar or function of wavelength
		The refractive index of the prism.
	orientation : scalar
		The orientation of the prism. The default orientation is aligned along the x-axis.
	'''
	def __init__(self, wedge_angle, refractive_index, orientation=0):
		self._wedge_angle = wedge_angle
		self._refractive_index = refractive_index
		self._orientation = orientation

		sag = lambda temp_grid : temp_grid.rotated(self._orientation).x * self._wedge_angle

		super().__init__(sag, self._refractive_index)

class Prism(SurfaceApodizer):
	'''A prism that deviates the beam.

	Parameters
	----------
	wedge_angle : scalar
		The wedge angle of the prism.
	refractive_index : scalar or function of wavelength
		The refractive index of the prism.
	orientation : scalar
		The orientation of the prism. The default orientation is aligned along the x-axis.
	'''
	def __init__(self, wedge_angle, refractive_index, orientation=0):
		self._wedge_angle = wedge_angle
		self._refractive_index = refractive_index
		self._orientation = orientation

		sag = lambda temp_grid : temp_grid.rotated(self._orientation).x * self._wedge_angle

		super().__init__(sag, self._refractive_index)

class PhaseGrating(PhaseApodizer):
	'''A grating that applies an achromatic phase pattern.

	Parameters
	----------
	grating_period : scalar
		The wedge angle of the prism.
	grating_amplitude : scalar
		The amplitude of the grating.
	grating_profile : Field or scalar or function
		The profile of the grating. The default is None and assumes a sinusoidal profile for the grating.
	orientation : scalar
		The orientation of the grating. The default orientation is aligned along the y-axis.
	'''
	def __init__(self, grating_period, grating_amplitude, grating_profile=None, orientation=0):
		self._grating_period = grating_period
		self._orientation = orientation

		if grating_profile is None:
			self._grating_profile = lambda temp_grid : grating_amplitude * np.sin(2 * np.pi * temp_grid.rotated(self._orientation) / self._grid_period)
		else:
			self._grating_profile = lambda temp_grid : grating_amplitude * grating_profile(temp_grid)

		super().__init__(self._grating_profile)