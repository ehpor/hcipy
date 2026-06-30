from .surface_profiles import parabolic_surface_sag
from .optical_element import OpticalElement
from .wavefront import Wavefront
import numpy as np

class ThinLens(OpticalElement):
    '''A parabolic thin lens.

    Parameters
    ----------
    focal_length : scalar
        The focal length of the thin lens at the reference wavelength.
    refractive_index : scalar or function of wavelength
        The refractive index of the lens material.
    reference_wavelength : scalar
        The wavelength for which the focal length is defined.
    '''
    def __init__(self, focal_length, refractive_index, reference_wavelength):
        self.reference_wavelength = reference_wavelength
        self.refractive_index = refractive_index
        self.focal_length = focal_length

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        else:
            return self.refractive_index

    @property
    def surface_sag(self):
        return parabolic_surface_sag(-self.radius_of_curvature)

    @property
    def radius_of_curvature(self):
        n0 = self._get_refractive_index(self.reference_wavelength)
        return self.focal_length * (n0 - 1)

    def forward(self, wavefront):
        surface_sag = self.surface_sag(wavefront.electric_field.grid)
        n = self._get_refractive_index(wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(1j * (n - 1) * surface_sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        surface_sag = self.surface_sag(wavefront.electric_field.grid)
        n = self._get_refractive_index(wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(-1j * (n - 1) * surface_sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)
