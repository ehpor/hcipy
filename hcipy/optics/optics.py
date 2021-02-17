import numpy as np
from .surface_profiles import parabolic_surface_sag
from .apodization import SurfaceApodizer

def make_thin_lens(focal_length, refractive_index, reference_wavelength):
	n0 = refractive_index(reference_wavelength)
	radius_of_curvature = focal_length * (n0 - 1)
	sag = parabolic_surface_sag(-radius_of_curvature)

	return SurfaceApodizer(sag, refractive_index)