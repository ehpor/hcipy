import numpy as np
from .optical_element import OpticalElement, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward

class Apodizer(AgnosticOpticalElement):
    '''A thin apodizer.

    This apodizer can apodize both in phase and amplitude.

    Parameters
    ----------
    apodization : Field or scalar or function of wavelength
        The apodization that we want to apply to any input wavefront.
    '''
    def __init__(self, apodization):
        self._apodization = apodization

        AgnosticOpticalElement.__init__(self, True, True)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        instance_data.apodization = self.evaluate_parameter(self.apodization, input_grid, output_grid, wavelength)

    @property
    def apodization(self):
        return self._apodization

    @apodization.setter
    def apodization(self, apodization):
        self._apodization = apodization

        self.clear_cache()

    def get_input_grid(self, output_grid, wavelength):
        return output_grid

    def get_output_grid(self, input_grid, wavelength):
        return input_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        wf = wavefront.copy()
        wf.electric_field *= instance_data.apodization

        return wf

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        wf = wavefront.copy()
        wf.electric_field *= instance_data.apodization.conj()

        return wf

class PhaseApodizer(Apodizer):
    '''A phase-only thin apodizer.

    Parameters
    ----------
    phase : Field or scalar or function
        The phase apodization.
    '''
    def __init__(self, phase):
        self._phase = phase

        Apodizer.__init__(self, self.apodization)

    @property
    def apodization(self):
        return self.construct_function(lambda p: np.exp(1j * p), self.phase)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

        self.clear_cache()

class SurfaceApodizer(Apodizer):
    '''A transmissive sagged surface optic.

    The surface is simulated as a thin plate. Propagation effects due to the
    thickness of the plate are not included. The supplied refractive index
    may change as function of wavelength.

    Parameters
    ----------
    surface_sag : Field or scalar or function
        The sag in the surface.
    refractive_index : scalar or function
        The refractive index of the material of the plate.
    '''
    def __init__(self, surface_sag, refractive_index):
        self._surface_sag = surface_sag
        self._refractive_index = refractive_index

        Apodizer.__init__(self, self.apodization)

    @property
    def opd(self):
        return self.construct_function(lambda n, surf: (n - 1) * surf, self.refractive_index, self.surface_sag)

    optical_path_difference = opd

    @property
    def phase(self):
        return self.construct_function(lambda opd, wavelength: opd * 2 * np.pi / wavelength, self.opd)

    @property
    def apodization(self):
        return self.construct_function(lambda p: np.exp(1j * p), self.phase)

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, refractive_index):
        self._refractive_index = refractive_index

        self.clear_cache()

    @property
    def surface_sag(self):
        return self._surface_sag

    @surface_sag.setter
    def surface_sag(self, surface_sag):
        self._surface_sag = surface_sag

        self.clear_cache()

class ComplexSurfaceApodizer(OpticalElement):
    def __init__(self, amplitude, surface, refractive_index):
        self.amplitude = amplitude
        self.surface = surface
        self.refractive_index = refractive_index

    def phase_for(self, wavelength):
        '''Get the phase screen in radians at a certain wavelength.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the phase screen.
        '''
        wavenumber = 2 * np.pi / wavelength

        opd = (self.refractive_index - 1) * self.surface

        return opd * wavenumber

    def forward(self, wavefront):
        opd = (self.refractive_index(wavefront.wavelength) - 1) * self.surface

        wf = wavefront.copy()
        wf.electric_field *= self.amplitude * np.exp(1j * opd * wf.wavenumber)

        return wf

    def backward(self, wavefront):
        opd = (self.refractive_index(wavefront.wavelength) - 1) * self.surface

        wf = wavefront.copy()
        wf.electric_field *= self.amplitude * np.exp(-1j * opd * wf.wavenumber)

        return wf

class MultiplexedComplexSurfaceApodizer(OpticalElement):
    def __init__(self, amplitude, surface, refractive_index):
        self.amplitude = amplitude
        self.surface = surface
        self.refractive_index = refractive_index

    def forward(self, wavefront):
        apodizer_mask = 0
        for amplitude, surface in zip(self.amplitude, self.surface):
            opd = (self.refractive_index(wavefront.wavelength) - 1) * surface
            apodizer_mask += amplitude * np.exp(1j * opd * wavefront.wavenumber)

        wf = wavefront.copy()
        wf.electric_field *= apodizer_mask
        return wf

    def backward(self, wavefront):
        apodizer_mask = 0
        for amplitude, surface in zip(self.amplitude, self.surface):
            opd = (self.refractive_index(wavefront.wavelength) - 1) * surface
            apodizer_mask += amplitude * np.exp(1j * opd * wavefront.wavenumber)

        wf = wavefront.copy()
        wf.electric_field /= apodizer_mask
        return wf
