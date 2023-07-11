from .surface_profiles import parabolic_surface_sag
from .apodization import SurfaceApodizer

class ThinLens(SurfaceApodizer):
    '''A parabolic thin lens.

    Parameters
    ----------
    focal_length : scalar
        The focal length of the thin lens at the refernce wavelength.
    refractive_index : scalar or function of wavelength
        The refractive index of the lens material.
    reference_wavelength : scalar
        The wavelength for which the focal length is defined.

    '''
    def __init__(self, focal_length, refractive_index, reference_wavelength):
        self._focal_length = focal_length
        self._refractive_index = refractive_index
        self._reference_wavelength = reference_wavelength

        n0 = refractive_index(reference_wavelength)
        radius_of_curvature = focal_length * (n0 - 1)
        sag = parabolic_surface_sag(-radius_of_curvature)
        super().__init__(sag, refractive_index)

    @property
    def focal_length(self):
        '''The focal length of the lens.
        '''
        return self._focal_length

    @focal_length.setter
    def focal_length(self, focal_length):
        self._focal_length = focal_length

        n0 = self._refractive_index(self._reference_wavelength)
        radius_of_curvature = focal_length * (n0 - 1)
        sag = parabolic_surface_sag(-radius_of_curvature)

        self.surface_sag = sag
