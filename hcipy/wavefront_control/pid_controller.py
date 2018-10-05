import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .calibration import calibrate_modal_reconstructor

class PIDController(object):
    '''A proportional integral derivate controller.

    This class implements an proportional integral derivate controller. 

    Parameters
    ----------
    proportional_gain : scalar
        Gain of the proportional term
    integral_gain: scalar
        Gain of the integral term
    derivative_gain: scalar
        Gain of the derivate term
    command_matrix : ndarray
        The command_matrix 
    reference : ndarray
        The reference the controller is convering to for its steady state.
    
    Attributes
    ----------
    actuators : ndarray
        The actuator commands to be given to the actuator (i.e., DM)
    '''
    def __init__(self, proportional_gain,integral_gain,derivative_gain,command_matrix=None, reference=0):
        self.error=0
        self.previous_error=0
        self.integrator=0
        self.derivative=0
        self.proportional_gain=proportional_gain
        self.integral_gain=integral_gain
        self.derivative_gain=derivative_gain
        self.reference = 0
        self.actuators = 0
        self.command_matrix = command_matrix
        self.x=[]


    @property
    def command_matrix(self):
        return self._command_matrix
    
    @command_matrix.setter
    def command_matrix(self, command_matrix):
        self._command_matrix = command_matrix

    @property
    def proportional_gain(self):
        return self._proportional_gain
    
    @proportional_gain.setter
    def proportional_gain(self, gain):
        self._proportional_gain = gain
    
    @property
    def integral_gain(self):
        return self._integral_gain
    
    @integral_gain.setter
    def integral_gain(self, gain):
        self._integral_gain = gain
    @property
    def derivative_gain(self):
        return self._derivative_gain
    
    @derivative_gain.setter
    def derivative_gain(self, gain):
        self._derivative_gain = gain

    def submit_wavefront(self, t, wavefront,  wfs_number=0):
        '''Submit a wavefront estimate to integrator.

        Parameters
        ----------
        t : scalar
            Time at which the estimate was taken.
        wavefront : Field
            The estimate of the wavefront. This can be slopes, mode coefficients, etc...
        wfs_number : int
            The index of the wavefront sensor. This is meant for support of multiple
            wavefront sensors.
        '''
        self.x=wavefront
    def get_actuators(self, t, dm_number=0):
        '''Get the actuator positions at time `t` for DM number `dm_number`.

        Parameters
        ----------
        t : scalar
            The time at which to get the requested actuator positions.
        dm_number : int
            The index of the deformable mirror. This is meant for support for multiple
            deformable mirrors.
        '''

        self.error = self.x - self.reference
        self.integrator-=self.error
        self.derivative=self.error-self.previous_error

        actuators =  self.proportional_gain * self.command_matrix.dot(self.error) +self.integral_gain * self.command_matrix.dot(self.integrator) +self.derivative_gain * self.command_matrix.dot(self.derivative)
        self.actuators=actuators
        self.previous_error=self.error
        return actuators