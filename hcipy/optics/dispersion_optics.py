from .optical_element import OpticalElement
from .apodization import SurfaceApodizer, PhaseApodizer
import numpy as np
from  ..field import Field, field_dot

def vector_snells_law(incidence_direction_cosine, surface_normal, relative_refractive_index):
	'''
	Parameters
	----------
	incidence_direction_cosine : Field
		The incidence angles of the incoming field.
	surface_normal : array-like
		The normal of the surface
	relative_refractive_index : scalar
		The relative refractive index between two media.

	Returns
	-------
	Array like
		The transmitted angles of the outgoing field.
	'''

	transmitted_normal_component = np.sqrt(1 - relative_refractive_index * field_dot(surface_normal, incidence_direction_cosine))
	incident_normal_component = field_dot(surface_normal, incidence_direction_cosine)[np.newaxis,:] * surface_normal[:, np.newaxis]
	transmitted_transverse_component = relative_refractive_index * (incidence_direction_cosine - incident_normal_component)

	return transmitted_normal_component + transmitted_transverse_component

def snells_law(incidence_angle, relative_refractive_index):
	''' Applies Snell's law.

	Parameters
	----------
	incidence_angle : array like
		The incidence angles of the incoming field.
	relative_refractive_index : scalar
		The relative refractive index between two media.

	Returns
	-------
	Array like
		The transmitted angles of the outgoing field.
	'''
	if relative_refractive_index > 1:
		return np.arcsin(relative_refractive_index * np.sin(incidence_angle))
	else:
		if np.all(incidence_angle < np.arcsin(relative_refractive_index)):
			return np.arcsin(relative_refractive_index * np.sin(incidence_angle))
		else:
			raise ValueError("Total internal reflection is occuring.")

class TiltElement(SurfaceApodizer):
	''' An element that applies a tilt.

	Parameters
	----------
	angle: scalar
		The tilt angle.
	orientation : scalar
		The orientation of the tilt. The default orientation is aligned along the y-axis.
	refractive_index : scalar or function
		The refractive index of the material. The default is 2.0 which makes it achromatic and exact.
	'''
	def __init__(self, angle, orientation, refractive_index = 2.0):
		self._angle = angle
		self._orientation = orientation
		
		sag = lambda temp_grid : Field(temp_grid.rotated(self._orientation).y * np.tan(self._angle), temp_grid)
		super().__init__(sag, refractive_index)
	
	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, new_angle):
		self._angle = new_angle
		self.surface_sag = lambda temp_grid : Field(temp_grid.rotated(self._orientation).y * np.tan(self._angle), temp_grid)

	@property
	def orientation(self):
		return self._orientation

	@orientation.setter
	def orientation(self, new_orientation):
		self._orientation = new_orientation
		self.surface_sag = lambda temp_grid : Field(temp_grid.rotated(self._orientation).y * np.tan(self._angle), temp_grid)

class ThinPrism(TiltElement):
	'''A thin prism that operates in the paraxial regime.

	Parameters
	----------
	wedge_angle : scalar
		The wedge angle of the prism.
	orientation : scalar
		The orientation of the prism. The default orientation is aligned along the x-axis.
	refractive_index : scalar or function of wavelength
		The refractive index of the prism.
	'''
	def __init__(self, wedge_angle, orientation, refractive_index):
		super().__init__(wedge_angle, orientation, refractive_index)
		

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
	def __init__(self, angle_of_incidence, prism_angle, refractive_index, orientation=0):
		self._prism_angle = prism_angle
		self._angle_of_incidence = angle_of_incidence
		self._refractive_index = refractive_index
		self._orientation = orientation

		super().__init__(self.prism_sag, self._refractive_index)
	
	def trace(self, wavelength):
		''' Trace a ray through the prism.
		
		Parameters
		----------
		wavelength : scalar
			The wavelength that is traced through the prism.

		Returns
		-------
		scalar
			The angle of deviation for the traced ray.
		'''
		n = self._refractive_index(wavelength)
		transmitted_angle_surface_1 = snells_law(self._angle_of_incidence, n)

		incident_angle_surface_2 = self._prism_angle - transmitted_angle_surface_1
		transmitted_angle = snells_law(incident_angle_surface_2, 1 / n)

		angle_of_deviation = self._angle_of_incidence + transmitted_angle - self._prism_angle
		
		return angle_of_deviation
	
	def prism_sag(self, grid, wavelength):
		''' Calculate the sag profile for the prism.
		'''
		theta = self.trace(wavelength)
		return Field(grid.rotated(self._orientation).y * np.tan(theta), grid)

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
			self._grating_profile = lambda temp_grid : grating_amplitude * grating_profile(temp_grid.scaled(1 / self._grid_period))

		super().__init__(self._grating_profile)