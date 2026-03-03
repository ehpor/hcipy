import numpy as np

from .optical_element import OpticalElement

class SimpleVibration(OpticalElement):
    '''A simple vibration model.

    Parameters
    ----------
    mode : Field
        The spatial mode of the vibration.
    amplitude : scalar
        The amplitude of the vibration. After multiplying the mode and
        the amplitude, the resuilting units should be meters.
    frequency : scalar
        The temporal frequency of the vibration in Hz.
    phase_0 : scalar, optional
        The initial phase of the vibration in radians. Default is 0.

    Attributes
    ----------
    t : float
        The time in seconds.
    '''
    def __init__(self, mode, amplitude, frequency, phase_0=0):
        self.mode = mode
        self.amplitude = amplitude
        self.phase_0 = phase_0
        self._frequency = frequency
        self.t = 0

    @property
    def frequency(self):
        '''The temporal frequency of the vibration in Hz.
        '''
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        delta_phase = 2 * np.pi * (self._frequency - frequency) * self.t
        self.phase_0 += delta_phase
        self._frequency = frequency

    @property
    def phase(self):
        '''The current phase of the vibration in radians.
        '''
        return 2 * np.pi * self.frequency * self.t + self.phase_0

    def forward(self, wavefront):
        '''Propagate the wavefront forward through the vibration.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront.

        Returns
        -------
        Wavefront
            The output wavefront with the vibration applied.

        '''
        wf = wavefront.copy()

        wf.electric_field *= np.exp(1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf

    def backward(self, wavefront):
        '''Propagate the wavefront backward through the vibration.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront.

        Returns
        -------
        Wavefront
            The output wavefront with the inverse vibration applied.

        '''
        wf = wavefront.copy()

        wf.electric_field *= np.exp(-1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
        return wf
