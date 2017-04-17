from .wavefront_sensor import WavefrontSensor, WavefrontSensorNew
from ..optics import *
from ..propagation import FraunhoferPropagator
from ..aperture import *
from ..field import *

import numpy as np

class FourierWavefrontSensor(OpticalSystem):
	def __init__(self, pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength, num_airy=None):
		# Create the intermediate and final grids
		output_grid = make_pupil_grid(output_grid_size, output_grid_pixels )
		focal_grid = make_focal_grid(pupil_grid, q = pupil_separation * q, num_airy=num_airy, wavelength=wavelength)

		# Make all the optical elements
		pupil_to_focal = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=wavelength)
		focal_mask = focal_plane_mask(focal_grid)
		focal_to_pupil = FraunhoferPropagator(focal_grid, output_grid, wavelength_0=wavelength)

		self.optical_elements = (pupil_to_focal, focal_mask, focal_to_pupil)

def pyramid_surface(refractive_index, separation, wavelength_0):
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(focal_grid.x) + np.abs(focal_grid.y))
		surf = Field(surf, focal_grid)
		return SurfaceApodizer(surf, refractive_index)
	return func

def pyramid_wavefront_sensor(pupil_grid, pupil_separation, num_pupil_pixels, q, wavelength, refractive_index, num_airy=None):
	# Make mask
	pupil_diameter = pupil_grid.x[pupil_mask(pupil_grid) != 0].ptp()
	sep = 0.5 * pupil_separation * pupil_diameter
	focal_plane_mask = pyramid_surface(refractive_index, sep)

	# Multiply by 2 because we want to have two pupils next to each other
	output_grid_size = 2 * pupil_separation * pupil_diameter
	output_grid_pixels = 2 * pupil_separation * num_pupil_pixels

	# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
	if q < 2:
		q = 2

	return FourierWavefrontSensor(pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength, num_airy)

# Achromatic zernike mask
def phase_step_mask(diameter=1, phase_step=np.pi/2):
	def func(grid):
		radius = focal_grid.as_('polar').r
		phase_mask = Field(phase_step * (radius <= diameter), focal_grid)
		return PhaseApodizer(surf, refractive_index)
	return func

def zernike_wavefront_sensor(pupil_grid, num_pupil_pixels, q, wavelength, diameter, num_airy=None):
	focal_plane_mask = phase_step_mask(diameter, np.pi/2)

	output_grid_size = pupil_grid.x[pupil_mask(pupil_grid) != 0].ptp()
	output_grid_pixels = num_pupil_pixels

	return FourierWavefrontSensor(pupil_grid, focal_plane_mask, output_grid_size, output_grid_pixels, q, wavelength, num_airy)

def vector_zernike_wavefront_sensor(pupil_grid, num_pupil_pixels, q, wavelength, diameter, num_airy=None):
	
	focal_plane_mask_right = phase_step_mask(diameter, np.pi/4)
	focal_plane_mask_left = phase_step_mask(diameter, -np.pi/4)

	output_grid_size = pupil_grid.x[pupil_mask(pupil_grid) != 0].ptp()
	output_grid_pixels = num_pupil_pixels

	optical_system_left_polarization = FourierWavefrontSensor(pupil_grid, focal_plane_mask_left, output_grid_size, output_grid_pixels, q, wavelength, num_airy)
	optical_system_right_polarization = FourierWavefrontSensor(pupil_grid, focal_plane_mask_right, output_grid_size, output_grid_pixels, q, wavelength, num_airy)
	return optical_system_left_polarization, optical_system_right_polarization