from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..optics import PhaseApodizer, LinearRetarder, Wavefront
from ..aperture import make_circular_aperture
from ..field import make_uniform_grid, Field

import numpy as np

class ZernikeWavefrontSensorOptics(WavefrontSensorOptics):
    '''The optical elements for a Zernike wavefront sensor.

    This class uses a propagation scheme of [NDiaye2013]_, similar to that of Lyot coronagraphs
    with a small blocking mask, see [Soummer2007]_.

    .. [NDiaye2013] N'Diaye et al. 2013, "Calibration of quasi-static
        aberrations in exoplanet direct-imaging instruments with a
        Zernike phase-mask sensor"

    .. [Soummer2007] Soummer et al. 2007, "Fast computation of Lyot-style
        coronagraph propagation"

    Parameters
    ----------
    input_grid : Grid
        The grid on which the input wavefront is defined.
    phase_step : scalar
        The phase of the phase dot of the zernike wavefront sensor. The default is pi/2.
    phase_dot_diameter : scalar
        The diameter of the phase dot. This has units of lambda_0/D.
    num_pix : scalar
        The number of pixels across the phase dot.
    pupil_diameter : scalar
        The diameter of the pupil. This is used for calculating the size of the phase dot.
    reference_wavelength : scalar
        The reference wavelength. This is used for calcualting the size of the phase dot.
    '''
    def __init__(self, input_grid, phase_step=np.pi / 2, phase_dot_diameter=1.06, num_pix=128, pupil_diameter=1, reference_wavelength=1):
        self.input_grid = input_grid
        self.output_grid = input_grid

        # Make the phase dot
        phase_dot_diameter *= reference_wavelength / pupil_diameter
        focal_grid = make_uniform_grid([num_pix, num_pix], phase_dot_diameter)
        self.phase_dot = PhaseApodizer(make_circular_aperture(phase_dot_diameter)(focal_grid) * phase_step)

        # Make the propagator
        self.prop = FraunhoferPropagator(input_grid, focal_grid)

    def forward(self, wavefront):
        '''Propagates a wavefront through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        '''
        wf_foc = self.prop.forward(wavefront)
        wf_foc.electric_field -= self.phase_dot.forward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = wavefront.electric_field - pup.electric_field

        return pup

    def backward(self, wavefront):
        '''Propagates a wavefront backwards through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        '''
        wf_foc = self.prop.forward(wavefront)
        wf_foc.electric_field -= self.phase_dot.backward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = wavefront.electric_field - pup.electric_field

        return pup

class ZernikeWavefrontSensorEstimator(WavefrontSensorEstimator):
    '''Estimates the wavefront slopes from pyramid wavefront sensor images.

    Parameters
    ----------
    aperture : function
        A function which mask the output phase measurements.
    output_grid : Grid
        The grid on which the output of a pyramid wavefront sensor is sampled.
    reference : Field
        A reference image to subtract from the Zernike wavefront sensor data.

    Attributes
    ----------
    measurement_grid : Grid
        The grid on which the phase measurements are defined.
    pupil_mask : array_like
        A mask for the phase measurements.
    num_measurements : int
        The number of pixels in the output vector.
    reference : Field
        A reference image to subtract from the Zernike wavefront sensor data.
    '''
    def __init__(self, aperture, output_grid, reference):
        self.measurement_grid = output_grid
        self.pupil_mask = aperture(self.measurement_grid)
        self.reference = reference
        self.num_measurements = int(np.sum(self.pupil_mask > 0))

    def estimate(self, images):
        '''A function which estimates the phase from a Zernike wavefront sensor image.

        Parameters
        ----------
        images - List
            A list of scalar intensity fields containing Zernike wavefront sensor images.

        Returns
        -------
        res - Field
            A field with phase estimates.
        '''
        image = images[0]

        intensity_measurements = (image - self.reference).ravel() * self.pupil_mask
        res = Field(intensity_measurements[self.pupil_mask > 0], self.pupil_mask.grid)
        return res

class VectorZernikeWavefrontSensorOptics(WavefrontSensorOptics):
    '''The optical elements for a vector-Zernike wavefront sensor.

    The vector Zernike is based on [Doelman2019]_.

    This class uses a propagation scheme of [NDiaye2013]_, similar to that of Lyot coronagraphs
    with a small blocking mask, see [Soummer2007]_. The reference wavefront is also propagated
    through a half-wave retarder because the vector-Zernike mask is a half-wave
    retarder.

    .. [Doelman2019] Doelman et al. 2019, "Simultaneous phase and amplitude aberration sensing
        with a liquid-crystal vector-Zernike phase mask"

    .. [NDiaye2013] N'Diaye et al. 2013, "Calibration of quasi-static
        aberrations in exoplanet direct-imaging instruments with a
        Zernike phase-mask sensor"

    .. [Soummer2007] Soummer et al. 2007, "Fast computation of Lyot-style
        coronagraph propagation"

    Parameters
    ----------
    input_grid : Grid
        The grid on which the input wavefront is defined.
    phase_retardation : scalar or Field
        The relative phase retardation induced between the fast and slow axis.
    phase_step : scalar
        The phase of the phase dot of the zernike wavefront sensor. The default is pi/2.
    phase_dot_diameter : scalar
        The diameter of the phase dot. This has units of lambda_0/D.
    num_pix : scalar
        The number of pixels across the phase dot.
    pupil_diameter : scalar
        The diameter of the pupil. This is used for calculating the size of the phase dot.
    reference_wavelength : scalar
        The reference wavelength. This is used for calcualting the size of the phase dot.
    '''
    def __init__(self, input_grid, phase_retardation=np.pi, phase_step=np.pi / 2, phase_dot_diameter=1.06, num_pix=128, pupil_diameter=1, reference_wavelength=1):
        self.input_grid = input_grid
        self.output_grid = input_grid
        self._phase_retardation = phase_retardation

        # Make the vector-Zernike wavefront sensor mask
        phase_dot_diameter *= reference_wavelength / pupil_diameter
        focal_grid = make_uniform_grid([num_pix, num_pix], phase_dot_diameter)
        phase_dot = make_circular_aperture(phase_dot_diameter)(focal_grid) * phase_step

        self.vZWFS_mask = LinearRetarder(phase_retardation, phase_dot / 2)

        # Make half-wave plate for reference offset
        self.HWP = LinearRetarder(phase_retardation, 0)

        # Make the propagator
        self.prop = FraunhoferPropagator(input_grid, focal_grid)

    @property
    def phase_retardation(self):
        '''The phase retardation of the linear retarder
        '''
        return self._phase_retardation

    @phase_retardation.setter
    def phase_retardation(self, phase_retardation):
        self._phase_retardation = phase_retardation

        self.vZWFS_mask.phase_retardation = phase_retardation
        self.HWP.phase_retardation = phase_retardation

    def forward(self, wavefront):
        '''Propagates a wavefront through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        '''
        wf_foc = self.prop.forward(wavefront)
        if wf_foc.is_scalar:
            wf_foc = Wavefront(wf_foc.electric_field, wavelength=wf_foc.wavelength, input_stokes_vector=(1, 0, 0, 0))

        wf_foc.electric_field = self.HWP.forward(wf_foc).electric_field - self.vZWFS_mask.forward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = self.HWP.forward(wavefront).electric_field - pup.electric_field

        return pup

    def backward(self, wavefront):
        '''Propagates a wavefront backwards through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        '''
        wf_foc = self.prop.forward(wavefront)
        if wf_foc.is_scalar:
            wf_foc = Wavefront(wf_foc.electric_field, wavelength=wf_foc.wavelength, input_stokes_vector=(1, 0, 0, 0))

        wf_foc.electric_field = self.HWP.backward(wf_foc).electric_field - self.vZWFS_mask.backward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = self.HWP.backward(wavefront).electric_field - pup.electric_field

        return pup
