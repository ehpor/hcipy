import numpy as np
from .surface_profiles import parabolic_surface_sag
from .apodization import SurfaceApodizer

def make_thin_lens(focal_length, refractive_index, reference_wavelength):
	'''Create a parabolic thin lens.

	Parameters
	----------
	focal_length : scalar
		The focal length of the thin lens.
	refractive_index : scalar or function of wavelength
		The refractive index of the lens material.
	reference_wavelength : scalar
		The wavelength for which the focal length is defined.
	
	Returns
	-------
	SurfaceApodizer
		The thin lens optical element.
	'''
	n0 = refractive_index(reference_wavelength)
	radius_of_curvature = focal_length * (n0 - 1)
	sag = parabolic_surface_sag(-radius_of_curvature)

	return SurfaceApodizer(sag, refractive_index)