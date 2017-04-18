from .wavefront_sensor import WavefrontSensorOptics
from ..propagation import FraunhoferPropagator
from ..optics import SurfaceApodizer, PhaseApodizer, MultiplexedComplexSurfaceApodizer
import numpy as np

class FourierWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength, num_airy=None):
		# Create the intermediate and final grids
		output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength)

		# Make all the optical elements
		pupil_to_focal = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=wavelength)
		focal_mask = focal_plane_mask(focal_grid)
		focal_to_pupil = FraunhoferPropagator(focal_grid, output_grid, wavelength_0=wavelength)

		self.optical_elements = (pupil_to_focal, focal_mask, focal_to_pupil)

def pyramid_surface(refractive_index, separation, wavelength_0):
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(grid.x) + np.abs(grid.y))
		surf = Field(surf, grid)
		return SurfaceApodizer(surf, refractive_index)
	return func

class PyramidWavefrontSensorOptics(FourierWavefrontSensorOptics):
	def __init__(self, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength, refractive_index, num_airy=None):
		# Make mask
		sep = 0.5 * pupil_separation * pupil_diameter
		focal_plane_mask = pyramid_surface(refractive_index, sep, wavelength)
		
		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = 2 * pupil_separation * pupil_diameter
		output_grid_pixels = np.ceil(2 * pupil_separation * num_pupil_pixels)

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation:
			q = 2 * pupil_separation

		FourierWavefrontSensorOptics.__init__(self, pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength, num_airy)

# Achromatic zernike mask
def phase_step_mask(diameter=1, phase_step=np.pi/2):
	def func(grid):
		radius = grid.as_('polar').r
		phase_mask = Field(phase_step * (radius <= diameter), grid)
		return PhaseApodizer(phase_mask)
	return func

class ZernikeWavefrontSensorOptics(FourierWavefrontSensorOptics):
	def __init__(self, pupil_grid, pupil_diameter, num_pupil_pixels, q, wavelength_0, diameter, num_airy=None):
		focal_plane_mask = phase_step_mask(diameter * wavelength_0, np.pi/2)

		output_grid_size = pupil_grid.x.ptp()
		output_grid_pixels = num_pupil_pixels

		FourierWavefrontSensor.__init__(self, pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength_0, num_airy)
'''
def vector_zernike_wavefront_sensor(pupil_grid, pupil_diameter, num_pupil_pixels, q, wavelength, diameter, num_airy=None):

	focal_plane_mask_right = phase_step_mask(diameter, np.pi/4)
	focal_plane_mask_left = phase_step_mask(diameter, -np.pi/4)

	output_grid_size = pupil_grid.x.ptp()
	output_grid_pixels = num_pupil_pixels

	optical_system_left_polarization = FourierWavefrontSensor(pupil_grid, focal_plane_mask_left, output_grid_size, output_grid_pixels, q, wavelength, num_airy)
	optical_system_right_polarization = FourierWavefrontSensor(pupil_grid, focal_plane_mask_right, output_grid_size, output_grid_pixels, q, wavelength, num_airy)
	return optical_system_left_polarization, optical_system_right_polarization
'''

def optical_differentiation_surface(amplitude_filter, separation, wavelength_0, refractive_index):
	def func(grid):
	
		surf1 = -separation / (refractive_index(wavelength_0) - 1) * (grid.x + grid.y)
		surf2 = -separation / (refractive_index(wavelength_0) - 1) * (-grid.x + grid.y)
		surf3 = -separation / (refractive_index(wavelength_0) - 1) * (grid.x + -grid.y)
		surf4 = -separation / (refractive_index(wavelength_0) - 1) * (-grid.x -grid.y)
		
		surf = ( Field(surf1, grid),
			Field(surf2, grid),
			Field(surf3, grid),
			Field(surf4, grid))

		amp = (Field(amplitude_filter(grid.x), grid),
			Field(amplitude_filter(-grid.x), grid),
			Field(amplitude_filter(grid.y), grid),
			Field(amplitude_filter(-grid.y), grid))

		return MultiplexedComplexSurfaceApodizer(amp, surf, refractive_index)
	return func

class OpticalDifferentiationWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy=None):
		# Make mask
		sep = 0.5 * pupil_separation * pupil_diameter
		focal_plane_mask = optical_differentiation_surface(amplitude_filter, sep, wavelength_0, refractive_index)
		
		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = 2 * pupil_separation * pupil_diameter
		output_grid_pixels = np.ceil(2 * pupil_separation * num_pupil_pixels)

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation:
			q = 2 * pupil_separation

		FourierWavefrontSensor.__init__(self, pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength_0, num_airy)

class RooftopWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	def __init__(self, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy=None):
		amplitude_filter = lambda x : (x<0)/np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)

class gODWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	def __init__(self, beta, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy=None):
		amplitude_filter = lambda x : ((x>0) * beta + (1 - beta)*(1+x)/2)/np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)

class PolgODWavefrontSensorOptics(OpticalDifferentiationWavefrontSensorOptics):
	def __init__(self, beta, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy=None):
		amplitude_filter = lambda x : np.cos( np.pi/2 * ((x>0) * beta + (1 - beta)*(1+x)/2) - np.pi/4 )/np.sqrt(2)
		OpticalDifferentiationWavefrontSensorOptics.__init__(self, amplitude_filter, pupil_grid, pupil_diameter, pupil_separation, num_pupil_pixels, q, wavelength_0, refractive_index, num_airy)
