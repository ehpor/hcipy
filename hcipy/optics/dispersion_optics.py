from ..dev import deprecated
from .optical_element import OpticalElement
from .wavefront import Wavefront
import numpy as np
from ..field import Field
from scipy.special import jv

def grating_equation(wavelength, order, period, angle_of_incidence):
    '''Calculate the angle of the diffracted beam.

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
    '''Calculate the diffraction efficiency from a sinusoidal phase grating.

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
    '''Apply Snell's law.

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

def _tilt_sag(orientation, angle):
    '''Calculate the sag profile for a tilt element.

    Parameters
    ----------
    orientation : scalar
        The orientation of the tilt in radians.
    angle : scalar
        The tilt angle in radians.

    Returns
    -------
    Field generator
        A function that takes a grid and returns the surface sag as a Field.
    '''
    def res(grid):
        return Field(grid.rotated(orientation).y * np.tan(angle), grid)

    return res

class TiltElement(OpticalElement):
    '''An element that applies a tilt.

    Parameters
    ----------
    angle: scalar
        The tilt angle in radians.
    orientation : scalar
        The orientation of the tilt in radians. The default orientation is aligned along the y-axis.
    refractive_index : scalar or function of wavelength
        The refractive index of the material. The default is 2.0 which makes it achromatic and exact.
    '''
    def __init__(self, angle, orientation=0, refractive_index=2.0):
        self.angle = angle
        self.orientation = orientation
        self.refractive_index = refractive_index

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        return self.refractive_index

    def forward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        sag = _tilt_sag(self.orientation, self.angle)(wavefront.electric_field.grid)

        new_field = wavefront.electric_field * np.exp(1j * (n - 1) * sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        sag = _tilt_sag(self.orientation, self.angle)(wavefront.electric_field.grid)

        new_field = wavefront.electric_field * np.exp(-1j * (n - 1) * sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

class ThinPrism(TiltElement):
    '''A thin prism that operates in the paraxial regime.

    Parameters
    ----------
    angle : scalar
        The wedge angle of the prism in radians.
    refractive_index : scalar or function of wavelength
        The refractive index of the prism.
    orientation : scalar
        The orientation of the prism in radians. The default orientation is aligned along the x-axis.
    '''
    def __init__(self, angle, refractive_index, orientation=0):
        super().__init__(angle, orientation, refractive_index)

    def minimal_deviation_angle(self, wavelength):
        '''Find the angle of minimal deviation for a paraxial prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of minimal deviation in radians.
        '''
        n = self._get_refractive_index(wavelength)
        return (n - 1) * self.angle

    def trace(self, wavelength):
        '''Trace a paraxial ray through the prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of deviation for the traced ray in radians.
        '''
        n = self._get_refractive_index(wavelength)
        return (n - 1) * self.angle

class Prism(OpticalElement):
    '''A prism that deviates the beam.

    Parameters
    ----------
    angle_of_incidence : scalar
        The angle of incidence of the wavefront in radians.
    prism_angle : scalar
        The angle of the prism in radians.
    refractive_index : scalar or function of wavelength
        The refractive index of the prism.
    orientation : scalar
        The orientation of the prism in radians. The default orientation is aligned along the y-axis.
    '''
    def __init__(self, angle_of_incidence, prism_angle, refractive_index, orientation=0):
        self.angle_of_incidence = angle_of_incidence
        self.prism_angle = prism_angle
        self.orientation = orientation
        self.refractive_index = refractive_index

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        return self.refractive_index

    def forward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        sag = self.prism_sag(wavefront.electric_field.grid, wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(1j * (n - 1) * sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        sag = self.prism_sag(wavefront.electric_field.grid, wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(-1j * (n - 1) * sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def minimal_deviation_angle(self, wavelength):
        '''Find the angle of minimal deviation for a prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of minimal deviation in radians.
        '''
        n = self._get_refractive_index(wavelength)
        return 2 * np.arcsin(n * np.sin(self.prism_angle / 2)) - self.prism_angle

    def trace(self, wavelength):
        '''Trace a ray through the prism.

        Parameters
        ----------
        wavelength : scalar
            The wavelength that is traced through the prism.

        Returns
        -------
        scalar
            The angle of deviation for the traced ray in radians.
        '''
        n = self._get_refractive_index(wavelength)
        return self._deviation(n)

    def prism_sag(self, grid, wavelength):
        '''Calculate the sag profile for the prism.

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
        n = self._get_refractive_index(wavelength)
        theta = self._deviation(n)
        sag = _tilt_sag(self.orientation, theta)(grid)
        return Field(sag / (n - 1), grid)

    def _deviation(self, n):
        '''Calculate the deviation angle for a given refractive index.
        '''
        transmitted_angle_surface_1 = snells_law(self.angle_of_incidence, 1 / n)
        incident_angle_surface_2 = self.prism_angle - transmitted_angle_surface_1
        transmitted_angle = snells_law(incident_angle_surface_2, n)

        return self.angle_of_incidence + transmitted_angle - self.prism_angle

class PhaseGrating(OpticalElement):
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
        self.grating_period = grating_period
        self.grating_amplitude = grating_amplitude
        self.orientation = orientation

        if grating_profile is None:
            def sinusoidal_grating_profile(grid):
                return np.sin(2 * np.pi * grid.y)

            grating_profile = sinusoidal_grating_profile

        self.grating_profile = grating_profile

    def grating_pattern(self, grid):
        return self.grating_amplitude * Field(self.grating_profile(grid.rotated(self.orientation).scaled(1 / self.grating_period)), grid)

    @property
    @deprecated('Use grating_period instead.')
    def period(self):
        return self.grating_period

    @period.setter
    @deprecated('Use grating_period instead.')
    def period(self, period):
        self.grating_period = period

    @property
    @deprecated('Use grating_amplitude instead.')
    def amplitude(self):
        return self.grating_amplitude

    @amplitude.setter
    @deprecated('Use grating_amplitude instead.')
    def amplitude(self, val):
        self.grating_amplitude = val

    def forward(self, wavefront):
        phase = self.grating_pattern(wavefront.electric_field.grid)
        new_field = wavefront.electric_field * np.exp(1j * phase)

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        phase = self.grating_pattern(wavefront.electric_field.grid)
        new_field = wavefront.electric_field * np.exp(-1j * phase)

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)
