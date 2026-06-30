import numpy as np
from .optical_element import OpticalElement
from .wavefront import Wavefront
import warnings

class Apodizer(OpticalElement):
    '''A thin apodizer.

    This apodizer can apodize both in phase and amplitude.

    Parameters
    ----------
    apodization : Field or scalar or function of wavelength
        The apodization that we want to apply to any input wavefront.
    '''
    def __init__(self, apodization):
        self.apodization = apodization

    def _get_apodization(self, wavelength):
        if callable(self.apodization):
            return self.apodization(wavelength)
        else:
            return self.apodization

    def forward(self, wavefront):
        a = self._get_apodization(wavefront.wavelength)
        new_field = wavefront.electric_field * a

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        a = self._get_apodization(wavefront.wavelength)
        new_field = wavefront.electric_field * np.conj(a)

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

class PhaseApodizer(OpticalElement):
    '''A phase-only thin apodizer.

    Parameters
    ----------
    phase : Field or scalar
        The phase apodization.
    '''
    def __init__(self, phase):
        self._phase = phase

    @property
    def phase(self):
        return self._phase

    def forward(self, wavefront):
        a = np.exp(1j * self.phase)
        new_field = wavefront.electric_field * a

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        a_conj = np.exp(-1j * self.phase)
        new_field = wavefront.electric_field * a_conj

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

class SurfaceApodizer(OpticalElement):
    '''A transmissive sagged surface optic.

    The surface is simulated as a thin plate. Propagation effects due to the
    thickness of the plate are not included. The supplied refractive index
    may change as function of wavelength.

    Parameters
    ----------
    surface_sag : Field
        The sag in the surface.
    refractive_index : scalar or function of wavelength
        The refractive index of the material of the plate.
    '''
    def __init__(self, surface_sag, refractive_index):
        self.surface_sag = surface_sag
        self.refractive_index = refractive_index

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        else:
            return self.refractive_index

    def forward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        a = np.exp(1j * (n - 1) * self.surface_sag * wavefront.wavenumber)

        new_field = wavefront.electric_field * a

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)
        a_conj = np.exp(-1j * (n - 1) * self.surface_sag * wavefront.wavenumber)

        new_field = wavefront.electric_field * a_conj

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

class ComplexSurfaceApodizer(OpticalElement):
    '''A surface apodizer with amplitude coating.

    Parameters
    ----------
    amplitude : Field or scalar or function of wavelength
        The amplitude apodization.
    surface : Field or scalar
        The sag in the surface.
    refractive_index : scalar or function of wavelength
        The refractive index of the material of the plate.
    '''
    def __init__(self, amplitude, surface, refractive_index):
        self.amplitude = amplitude
        self.surface = surface
        self.refractive_index = refractive_index

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        else:
            return self.refractive_index

    def _get_amplitude(self, wavelength):
        if callable(self.amplitude):
            return self.amplitude(wavelength)
        else:
            return self.amplitude

    def forward(self, wavefront):
        amp = self._get_amplitude(wavefront.wavelength)
        n = self._get_refractive_index(wavefront.wavelength)

        a = amp * np.exp(1j * (n - 1) * self.surface * wavefront.wavenumber)
        new_field = wavefront.electric_field * a

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        amp_conj = np.conj(self._get_amplitude(wavefront.wavelength))
        n = self._get_refractive_index(wavefront.wavelength)

        a_conj = amp_conj * np.exp(-1j * (n - 1) * self.surface * wavefront.wavenumber)
        new_field = wavefront.electric_field * a_conj

        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

class MultiplexedComplexSurfaceApodizer(OpticalElement):
    '''A non-physical apodizer that consists of multiple :class:`ComplexSurfaceApodizer`
    apodizers on top of each other.

    Parameters
    ----------
    amplitudes : list of {Field or scalars or functions of wavelength}
        The amplitude apodizations of each of the masks.
    surfaces : list of Fields
        The surface sags of each of the masks.
    refractive_index : scalar or function of wavelength
        The refractive index of the material of the plate.
    '''
    def __init__(self, amplitudes, surfaces, refractive_index, amplitude=None, surface=None):
        if amplitude is not None:
            warnings.warn("The `amplitude` parameter has been deprecated and will be removed in a future version. Use `amplitudes` instead.", DeprecationWarning, stacklevel=2)
            amplitudes = amplitude

        if surface is not None:
            warnings.warn("The `surface` parameter has been deprecated and will be removed in a future version. Use `surfaces` instead.", DeprecationWarning, stacklevel=2)
            surfaces = surface

        self.amplitudes = amplitudes
        self.surfaces = surfaces
        self.refractive_index = refractive_index

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        else:
            return self.refractive_index

    @staticmethod
    def _get_amplitude(amplitude, wavelength):
        if callable(amplitude):
            return amplitude(wavelength)
        else:
            return amplitude

    def forward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)

        apodizer_mask = 0
        for amplitude, surface in zip(self.amplitudes, self.surfaces):
            amp = self._get_amplitude(amplitude, wavefront.wavelength)
            apodizer_mask += amp * np.exp(1j * (n - 1) * surface * wavefront.wavenumber)

        new_field = wavefront.electric_field * apodizer_mask
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        n = self._get_refractive_index(wavefront.wavelength)

        apodizer_mask_conj = 0
        for amplitude, surface in zip(self.amplitudes, self.surfaces):
            amp_conj = np.conj(self._get_amplitude(amplitude, wavefront.wavelength))
            apodizer_mask_conj += amp_conj * np.exp(-1j * (n - 1) * surface * wavefront.wavenumber)

        new_field = wavefront.electric_field * apodizer_mask_conj
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)
