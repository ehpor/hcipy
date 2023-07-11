from .apodization import SurfaceApodizer, PhaseApodizer
import numpy as np
from ..field import Field
from scipy.special import jv

def grating_equation(wavelength, order, period, angle_of_incidence):
    ''' Calculates the angle of the diffracted beam.

    Parameters
    ----------
    wavelength : array like
        The wavelength for which the diffraction efficiency is calculated.
    order : array like
        The diffraction order.
    period : array like
        The period of the diffraction grating.
    angle_of_incidence : array like
        The incidence angles of the incoming field in radians.

    Returns
    -------
    Array like
        The diffraction efficiency
    '''
    return np.arcsin(order * wavelength / period + np.sin(angle_of_incidence))

def diffraction_efficiency_sinusoidal_grating(wavelength, groove_depth, order, period, angle_of_incidence):
    ''' Calculates the diffraction efficiency from a sinusoidal phase grating.

    Parameters
    ----------
    wavelength : array like
        The wavelength for which the diffraction efficiency is calculated.
    groove_depth : arraylike
        The depth of the sinusoidal grating.
    order : array like
        The diffraction order.
    period : array like
        The period of the diffraction grating.
    angle_of_incidence : array like
        The incidence angles of the incoming field in radians.

    Returns
    -------
    Array like
        The diffraction efficiency
    '''
    diffracted_angle = grating_equation(wavelength, order, period, angle_of_incidence)
    phase_difference = np.pi * groove_depth / wavelength * (np.cos(angle_of_incidence) + np.cos(diffracted_angle))
    return jv(order, phase_difference)**2

def snells_law(incidence_angle, relative_refractive_index):
    ''' Applies Snell's law.

    Parameters
    ----------
    incidence_angle : array like
        The incidence angles of the incoming field in radians.
    relative_refractive_index : scalar
        The relative refractive index between two media.

    Returns
    -------
    Array like
        The transmitted angles of the outgoing field in radians.
    '''
    if np.all(relative_refractive_index > 1):
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
        The tilt angle in radians.
    orientation : scalar
        The orientation of the tilt in radians. The default orientation is aligned along the y-axis.
    refractive_index : scalar or function
        The refractive index of the material. The default is 2.0 which makes it achromatic and exact.
    '''
    def __init__(self, angle, orientation=0, refractive_index=2.0):
        self._angle = angle
        self._orientation = orientation
        super().__init__(self.tilt_sag, refractive_index)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, new_angle):
        self._angle = new_angle
        self.surface_sag = self.tilt_sag

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation = new_orientation
        self.surface_sag = self.tilt_sag

    def tilt_sag(self, grid):
        ''' Calculate the sag profile for the tilt element.

        Parameters
        ----------
        grid : Grid
            The grid on which the surface sag is calculated.

        Returns
        -------
        Field
            The surface sag.
        '''
        return Field(grid.rotated(self._orientation).y * np.tan(self._angle), grid)

class ThinPrism(TiltElement):
    '''A thin prism that operates in the paraxial regime.

    Parameters
    ----------
    angle : scalar
        The wedge angle of the prism in radians.
    orientation : scalar
        The orientation of the prism in radians. The default orientation is aligned along the x-axis.
    refractive_index : scalar or function of wavelength
        The refractive index of the prism.
    '''
    def __init__(self, angle, refractive_index, orientation=0):
        super().__init__(angle, orientation, refractive_index)

    def minimal_deviation_angle(self, wavelength):
        ''' Find the angle of minimal deviation for a paraxial prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of minimal deviation in radians.
        '''
        n = self._refractive_index(wavelength)
        return (n - 1) * self._prism_angle

    def trace(self, wavelength):
        ''' Trace a paraxial ray through the prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of deviation for the traced ray in radians.
        '''
        return (self._refractive_index(wavelength) - 1) * self.angle

class Prism(SurfaceApodizer):
    '''A prism that deviates the beam.

    Parameters
    ----------
    angle_of_incidence : sacalar
        The angle of incidence of the wavefront in radians.
    prism_angle : scalar
        The angle of the prism in radians.
    refractive_index : scalar or function of wavelength
        The refractive index of the prism.
    orientation : scalar
        The orientation of the prism in radians. The default orientation is aligned along the y-axis.
    '''
    def __init__(self, angle_of_incidence, prism_angle, refractive_index, orientation=0):
        self._prism_angle = prism_angle
        self._angle_of_incidence = angle_of_incidence
        self._refractive_index = refractive_index
        self._orientation = orientation

        super().__init__(self.prism_sag, self._refractive_index)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation = new_orientation
        self.surface_sag = self.prism_sag

    @property
    def prism_angle(self):
        return self._prism_angle

    @prism_angle.setter
    def prism_angle(self, new_prism_angle):
        self._prism_angle = new_prism_angle
        self.surface_sag = self.prism_sag

    def minimal_deviation_angle(self, wavelength):
        ''' Find the angle of minimal deviation for a prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of minimal deviation in radians.
        '''
        n = self._refractive_index(wavelength)
        return 2 * np.arcsin(n * np.sin(self._prism_angle / 2)) - self._prism_angle

    def trace(self, wavelength):
        ''' Trace a ray through the prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of deviation for the traced ray in radians.
        '''
        n = self._refractive_index(wavelength)
        transmitted_angle_surface_1 = snells_law(self._angle_of_incidence, 1 / n)

        incident_angle_surface_2 = self._prism_angle - transmitted_angle_surface_1
        transmitted_angle = snells_law(incident_angle_surface_2, n)

        angle_of_deviation = self._angle_of_incidence + transmitted_angle - self._prism_angle

        return angle_of_deviation

    def prism_sag(self, grid, wavelength):
        ''' Calculate the sag profile for the prism.

        Parameters
        ----------
        grid : Grid
            The grid on which the surface sag is calculated.
        wavelength : scalar
            The wavelength for which the surface sag is calculated.

        Returns
        -------
        Field
            The surface sag.
        '''
        theta = self.trace(wavelength)
        return Field(grid.rotated(self._orientation).y * np.tan(theta) / (self._refractive_index(wavelength) - 1), grid)

class PhaseGrating(PhaseApodizer):
    '''A grating that applies an achromatic phase pattern.

    Parameters
    ----------
    grating_period : scalar
        The period of the grating.
    grating_amplitude : scalar
        The amplitude of the grating.
    grating_profile : field generator
        The profile of the grating. The default is None and assumes a sinusoidal profile for the grating.
    orientation : scalar
        The orientation of the grating in radians. The default orientation is aligned along the y-axis.
    '''
    def __init__(self, grating_period, grating_amplitude, grating_profile=None, orientation=0):
        self._grating_period = grating_period
        self._orientation = orientation
        self._grating_amplitude = grating_amplitude

        if grating_profile is None:
            def sinusoidal_grating_profile(grid):
                return np.sin(2 * np.pi * grid.y)

            grating_profile = sinusoidal_grating_profile

        self._grating_profile = grating_profile

        super().__init__(self.grating_pattern)

    def grating_pattern(self, grid):
        return self._grating_amplitude * Field(self._grating_profile(grid.rotated(self._orientation).scaled(1 / self._grating_period)), grid)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation = new_orientation
        self.phase = self.grating_pattern

    @property
    def period(self):
        return self._grating_period

    @period.setter
    def period(self, new_period):
        self._grating_period = new_period
        self.phase = self.grating_pattern

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self._amplitude = new_amplitude
        self.phase = self.grating_pattern
