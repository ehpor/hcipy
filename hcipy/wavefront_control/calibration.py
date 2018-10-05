import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated

def make_command_matrix( system_function, amplitude=0.005,ref_commands=0):
    '''Simple calibration for AO system

    This function is a simple calibrator (linear and time-invariant) that finds the command matrix for a given AO system. 
    The optical system is completely user defined and this function is unaware of the spefic configuration. 

    Parameters
    ----------
    system_function : function
        A function that contains the desired AO system. It should take actuator commands and give back the desired modes
    
    amplitude : scalar
        The actuator poke 
    
    ref_commands: ndarray
        The actuator commands to provide the desired reference shape (potentially non-common path corrections) which we want to linearize our system around. 
    Return
    ----------
    command_matrix : ndarray
        The command matrix for the specified AO system given by the system_function. 
    '''
    ref=system_function(ref_commands)

    for i in range(max(ref.shape)):
        total_slopes = np.zeros(ref.shape)

        for amp in np.array([-1*amplitude, amplitude]):    
            slopes = system_function(amp)
            total_slopes += (slopes - ref) / (2 * amp)
        influence_functions.append(f.estimate(total_slopes, 0))
        
    influence_functions = ModeBasis(influence_functions)
    command_matrix = inverse_truncated(influence_functions.transformation_matrix)

    return command_matrix
