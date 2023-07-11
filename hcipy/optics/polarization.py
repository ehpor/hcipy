import numpy as np

from .wavefront import Wavefront, jones_to_mueller
from .optical_element import OpticalElement, make_agnostic_forward, make_agnostic_backward, AgnosticOpticalElement
from ..field import Field, field_dot, field_conjugate_transpose

def rotation_matrix(angle):
    '''Two dimensional rotation matrix.

    Parameters
    ----------
    angle : scaler
        rotation angle in radians

    Returns
    -------
    ndarray
        The rotation matrix.
    '''
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

class JonesMatrixOpticalElement(AgnosticOpticalElement):
    '''A general Jones Matrix.

    Parameters
    ----------
    jones_matrix : matrix or Field
        The Jones matrix describing the optical element.
    '''
    def __init__(self, jones_matrix):
        self.jones_matrix = jones_matrix

        AgnosticOpticalElement.__init__(self, True, True)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        instance_data.jones_matrix = self.evaluate_parameter(self.jones_matrix, input_grid, output_grid, wavelength)

    def get_input_grid(self, output_grid, wavelength):
        return output_grid

    def get_output_grid(self, input_grid, wavelength):
        return input_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        '''Propgate the wavefront through the Jones matrix.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        Wavefront
            The propagated wavefront.
        '''
        if wavefront.is_scalar:
            # We generate an unpolarized wavefront
            if hasattr(instance_data.jones_matrix, 'grid'):
                electric_field = instance_data.jones_matrix * wavefront.electric_field
            else:
                electric_field = instance_data.jones_matrix[..., np.newaxis] * wavefront.electric_field

            return Wavefront(electric_field, wavelength=wavefront.wavelength, input_stokes_vector=[1, 0, 0, 0])
        else:
            wf = wavefront.copy()
            wf.electric_field = field_dot(instance_data.jones_matrix, wf.electric_field)

            return wf

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        '''Propagate the wavefront backwards through the Jones matrix.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        Wavefront
            The propagated wavefront.
        '''
        if wavefront.is_scalar:
            # We generate an unpolarized wavefront
            if hasattr(instance_data.jones_matrix, 'grid'):
                electric_field = field_conjugate_transpose(instance_data.jones_matrix) * wavefront.electric_field
            else:
                electric_field = field_conjugate_transpose(instance_data.jones_matrix)[..., np.newaxis] * wavefront.electric_field

            return Wavefront(electric_field, wavelength=wavefront.wavelength, input_stokes_vector=[1, 0, 0, 0])
        else:
            wf = wavefront.copy()
            wf.electric_field = field_dot(field_conjugate_transpose(instance_data.jones_matrix), wf.electric_field)

            return wf

    @property
    def mueller_matrix(self):
        '''Returns the Mueller matrix corresponding to the Jones matrix.
        '''
        return self.construct_function(jones_to_mueller, self.jones_matrix)

    def __mul__(self, b):
        '''Multiply the Jones matrix on the right side with the Jones matrix m.

        Parameters
        ----------
        m : JonesMatrixOpticalElement, Field or matrix
            The Jones matrix with which to multiply.
        '''
        if hasattr(b, 'jones_matrix'):
            jones_matrix = self.construct_function(field_dot, self.jones_matrix, b.jones_matrix)
        else:
            jones_matrix = self.construct_function(field_dot, self.jones_matrix, b)

        return JonesMatrixOpticalElement(jones_matrix)

class PhaseRetarder(JonesMatrixOpticalElement):
    '''A general phase retarder.

    Parameters
    ----------
    phase_retardation : scalar or Field
        The relative phase retardation induced between the fast and slow axis.
    fast_axis_orientation : scalar or Field
        The angle of the fast axis with respect to the x-axis in radians.
    circularity : scalar or Field
        The circularity of the phase retarder.
    '''
    def __init__(self, phase_retardation, fast_axis_orientation, circularity):
        self._phase_retardation = phase_retardation
        self._fast_axis_orientation = fast_axis_orientation
        self._circularity = circularity

        JonesMatrixOpticalElement.__init__(self, self.jones_matrix)

    @property
    def jones_matrix(self):
        def jones(phase_retardation, fast_axis_orientation, circularity):
            phi_plus = np.exp(1j * phase_retardation / 2)
            phi_minus = np.exp(-1j * phase_retardation / 2)

            # calculating the individual components
            j11 = phi_plus * np.cos(fast_axis_orientation)**2 + phi_minus * np.sin(fast_axis_orientation)**2
            j12 = (phi_plus - phi_minus) * np.exp(-1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
            j21 = (phi_plus - phi_minus) * np.exp(1j * circularity) * np.cos(fast_axis_orientation) * np.sin(fast_axis_orientation)
            j22 = phi_plus * np.sin(fast_axis_orientation)**2 + phi_minus * np.cos(fast_axis_orientation)**2

            # constructing the Jones matrix.
            jones_matrix = np.array([[j11, j12], [j21, j22]])

            if hasattr(j11, 'grid'):
                jones_matrix = Field(jones_matrix, j11.grid)

            return jones_matrix

        return self.construct_function(jones, self.phase_retardation, self.fast_axis_orientation, self.circularity)

    @jones_matrix.setter
    def jones_matrix(self, b):
        pass

    @property
    def phase_retardation(self):
        return self._phase_retardation

    @phase_retardation.setter
    def phase_retardation(self, phase_retardation):
        self._phase_retardation = phase_retardation

        self.clear_cache()

    @property
    def fast_axis_orientation(self):
        return self._fast_axis_orientation

    @fast_axis_orientation.setter
    def fast_axis_orientation(self, fast_axis_orientation):
        self._fast_axis_orientation = fast_axis_orientation

        self.clear_cache()

    @property
    def circularity(self):
        return self._circularity

    @circularity.setter
    def circularity(self, circularity):
        self._circularity = circularity

        self.clear_cache()

class LinearRetarder(PhaseRetarder):
    '''A general linear retarder.

    Parameters
    ----------
    phase_retardation : scalar or Field
        The relative phase retardation induced between the fast and slow axis.
    fast_axis_orientation : scalar or Field
        The angle of the fast axis with respect to the x-axis in radians.
    '''
    def __init__(self, phase_retardation, fast_axis_orientation):
        PhaseRetarder.__init__(self, phase_retardation, fast_axis_orientation, 0)

class CircularRetarder(PhaseRetarder):
    '''A general circular retarder.

    Parameters
    ----------
    phase_retardation : scalar or Field
        The relative phase retardation induced between the fast and slow axis.
    fast_axis_orientation : scalar or Field
        The angle of the fast axis with respect to the x-axis in radians.
    '''
    def __init__(self, phase_retardation):
        PhaseRetarder.__init__(self, phase_retardation, np.pi / 4, np.pi / 2)

class QuarterWavePlate(LinearRetarder):
    '''A quarter-wave plate.

    Parameters
    ----------
    fast_axis_orientation : scalar or Field
        The angle of the fast axis with respect to the x-axis in radians.
    '''
    def __init__(self, fast_axis_orientation):
        LinearRetarder.__init__(self, np.pi / 2, fast_axis_orientation)

class HalfWavePlate(LinearRetarder):
    '''A half-wave plate.

    Parameters
    ----------
    fast_axis_orientation : scalar or Field
        The angle of the fast axis with respect to the x-axis in radians.
    '''
    def __init__(self, fast_axis_orientation):
        LinearRetarder.__init__(self, np.pi, fast_axis_orientation)

class GeometricPhaseElement(LinearRetarder):
    '''A general geometric phase element.

    Parameters
    ----------
    phase_pattern : Field or array_like
        The phase pattern in radians.
    leakage : scalar
        The relative leakage strength (0 = no leakage, 1 = maximum leakage)
    retardance_offset : scalar
        The retardance offset from half wave in radians. This will result in leakage.
    '''
    def __init__(self, phase_pattern, leakage=None, retardance_offset=0):
        if leakage is not None:
            # Testing if the leakage is valid.
            if leakage < 0 or leakage > 1:
                raise ValueError('Leakage must be between 0 and 1.')

            # Calculating the required retardance offset to get the leakage.
            retardance_offset = 2 * np.arcsin(np.sqrt(leakage))

        LinearRetarder.__init__(self, np.pi - retardance_offset, phase_pattern / 2)

class LinearPolarizer(JonesMatrixOpticalElement):
    '''A linear polarizer.

    Parameters
    ----------
    polarization_angle : scalar or Field
        The polarization angle of the polarizer. Light with this angle is transmitted.
    '''
    def __init__(self, polarization_angle):
        self.polarization_angle = polarization_angle

        JonesMatrixOpticalElement.__init__(self, self.jones_matrix)

    @property
    def jones_matrix(self):
        def jones(angle):
            c = np.cos(angle)
            s = np.sin(angle)

            return np.array([[c**2, c * s], [c * s, s**2]])

        return self.construct_function(jones, self.polarization_angle)

    @jones_matrix.setter
    def jones_matrix(self, b):
        pass

    @property
    def polarization_angle(self):
        '''The angle of polarization of the linear polarizer.
        '''
        return self._polarization_angle

    @polarization_angle.setter
    def polarization_angle(self, angle):
        self._polarization_angle = angle

        self.clear_cache()

class LinearPolarizingBeamSplitter(OpticalElement):
    ''' A linear polarizing beam splitter that accepts one wavefront and returns two orthogonally linearly polarized wavefronts.

    The first of the returned wavefront will have the polarization state set by the polarization angle. The second wavefront
    will be perpendicular to the first.

    Parameters
    ----------
    polarization_angle : scalar or Field
        The polarization angle of the polarizer.
    '''
    def __init__(self, polarization_angle, wavelength=1):
        self.polarization_angle = polarization_angle

    @property
    def polarization_angle(self):
        '''The angle of polarization of the linear polarizer.
        '''
        return self._polarization_angle

    @polarization_angle.setter
    def polarization_angle(self, polarization_angle):
        self._polarization_angle = polarization_angle

        # Calculating two Jones matrices for the two ports.
        self.polarizer_port_1 = LinearPolarizer(polarization_angle)
        self.polarizer_port_2 = LinearPolarizer(polarization_angle + np.pi / 2)

    def forward(self, wavefront):
        '''Propgate the wavefront through the PBS.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        wf_1 : Wavefront
            The wavefront propagated through a polarizer under the polarization angle.

        wf_2 : Wavefront
            The propagated wavefront through a polarizer perpendicular to the first one.
        '''

        wf_1 = self.polarizer_port_1.forward(wavefront)
        wf_2 = self.polarizer_port_2.forward(wavefront)

        return wf_1, wf_2

    def backward(self, wavefront):
        '''Propagate the wavefront backwards through the PBS.

        Not possible, will raise error.
        '''
        raise RuntimeError('Backward propagation through PolarizingBeamSplitter not possible.')

    @property
    def mueller_matrices(self):
        '''Returns the Mueller matrices of the two Jones matrices.
        '''
        return jones_to_mueller(self.polarizer_port_1.jones_matrix), jones_to_mueller(self.polarizer_port_2.jones_matrix)

class CircularPolarizingBeamSplitter(OpticalElement):
    ''' A circular polarizing beam splitter that accepts one wavefront and returns two linearly polarized wavefronts.

    The circular polarizing beam splitter is a combination of a quarter-wave plate and a polarizing beam splitter. The
    quarter-wave plate rotation angle is 45 degrees and the polarizing beam-splitter rotation angle is zero degrees.
    Therefore, the first of the returned wavefronts will have a linear polarization state with an electric field equal
    to the amount of left-circular polarization in the incoming wavefront. The second wavefront will be perpendicular
    to the first and with electric field equal to the amount of right-circular polarization in the incoming wavefront.

    Parameters
    ----------
    polarization_angle : scalar or Field
        The polarization angle of the polarizer.
    '''
    def __init__(self, wavelength=1):
        self.polarization_angle = 0
        self.quarter_wave_angle = np.pi / 4
        self.quarter_wave_plate = QuarterWavePlate(self.quarter_wave_angle)
        self.linear_polarizing_beam_splitter = LinearPolarizingBeamSplitter(self.polarization_angle)

    def forward(self, wavefront):
        '''Propgate the wavefront through the CBS.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        wf_1 : Wavefront
            The wavefront propagated through a quarter-wave plate at 45 degrees and a polarizer 0 degrees.
        wf_2 : Wavefront
            The wavefront propagated through a quarter-wave plate at 45 degrees and a polarizer 90 degrees.
        '''
        wf_1, wf_2 = self.linear_polarizing_beam_splitter.forward(self.quarter_wave_plate.forward(wavefront))
        return wf_1, wf_2

    def backward(self, wavefront):
        '''Propagate the wavefront backwards through the CBS.

        Not possible, will raise error.
        '''
        raise RuntimeError('Backward propagation through PolarizingBeamSplitter not possible.')

    @property
    def mueller_matrices(self):
        '''Returns the Mueller matrices of the two Jones matrices.
        '''
        qwp_mueller = self.quarter_wave_plate.mueller_matrix
        lin_pol_bs_mueller_1, lin_pol_bs_mueller_2 = self.linear_polarizing_beam_splitter.mueller_matrices

        return np.dot(lin_pol_bs_mueller_1, qwp_mueller), np.dot(lin_pol_bs_mueller_2, qwp_mueller)
