import numpy as np
from ..optics import OpticalElement, PhaseApodizer, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, make_focal_grid, Field, CartesianGrid, RegularCoords
from ..plotting import imshow_field
from ..mode_basis import make_gaussian_hermite_basis, make_LP_modes

class FiberNuller(OpticalElement):
    '''A generic fiber nuller

    Parameters
    ----------
    input_grid: Grid
        The grid on which the incoming wavefront is defined.
    mode_field_diameter: scalar
        The mode field diameter for the gaussian approximation of a fiber.
    pupil_diameter: scalar
        The pupil diameter.
    focal_length: scalar
        The focal length for injection into the fiber.
    wavelength: scalar
        The wavelength of the simulation.
    phase_screen: Field
        The pupil plane phase screen, defined on input_grid.
    '''
    def __init__(self, input_grid, mode_field_diameter=1.4, pupil_diameter=1,focal_length=1, wavelength=1, phase_screen=None):
        self.input_grid = input_grid
        self.mode_field_diameter = mode_field_diameter
        self.pupil_diameter = pupil_diameter
        self.focal_length = focal_length
        self.wavelength = wavelength

        if phase_screen is not None:
            self.phase_screen = phase_screen
        else:
            phase_screen = np.zeros(input_grid.shape).ravel()

        self.focal_grid = make_focal_grid(q=11,num_airy=3,pupil_diameter=self.pupil_diameter,focal_length=self.focal_length,reference_wavelength=self.wavelength)
        self.prop = FraunhoferPropagator(self.input_grid, self.focal_grid)
        self.phase_screen_optic = PhaseApodizer(phase_screen)

        fiber_mode = make_gaussian_hermite_basis(self.focal_grid, 1, self.mode_field_diameter)[0]
        fiber_mode /= np.sqrt(np.sum(np.abs(fiber_mode)**2 * self.focal_grid.weights))
        self.fiber_mode = fiber_mode
        self.output_grid = CartesianGrid(RegularCoords([1, 1], [1, 1], np.zeros(2)))
        self.output_grid.weights = 1

    def forward(self, wavefront):
        '''Propagate a wavefront through the fiber nuller

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is expected to be
            in the pupil plane.

        Returns
        -------
        Field
            The coupling amplitude through the fiber nuller as a complex scalar.
        '''
        wavelength = wavefront.wavelength

        post_phase_wf = self.phase_screen_optic.forward(wavefront)
        foc = self.prop.forward(post_phase_wf)

        output = np.dot(foc.electric_field.conj() * self.focal_grid.weights, self.fiber_mode)
        output = Field(output, self.output_grid)

        return output

    def backward(self, wavefront):
        '''Propagate backwards through the fiber nuller

        Parameters
        ----------
        wavefront : scalar
            The wavefront to propagate. This wavefront is expected to be
            a complex scalar representing light emerging from the fiber.

        Returns
        -------
        Wavefront
            The pupil plane wavefront.
        '''

        fib = self.fiber_mode * wavefront
        foc = Wavefront(fib,wavelength=self.wavelength)
        collimated = self.prop.backward(foc)
        output = self.phase_screen_optic.backward(collimated)

        return output

class VortexFiberNuller(FiberNuller):
    '''A vortex fiber nuller

    Parameters
    ----------
    input_grid: Grid
        The grid on which the incoming wavefront is defined.
    mode_field_diameter: scalar
        The mode field diameter for the gaussian approximation of a fiber.
    pupil_diameter: scalar
        The pupil diameter.
    focal_length: scalar
        The focal length for injection into the fiber.
    wavelength: scalar
        The wavelength of the simulation.
    vortex_charge: integer
        The vortex charge.
    '''
    def __init__(self, input_grid, mode_field_diameter=1.4, pupil_diameter=1,focal_length=1, wavelength=1, vortex_charge=1):
        phase_screen_gen = lambda grid: Field(vortex_charge * grid.as_('polar').theta, grid)
        phase_screen = phase_screen_gen(input_grid)
        super().__init__(input_grid,mode_field_diameter,pupil_diameter,focal_length,wavelength,phase_screen)

class PhotonicLanternNuller(FiberNuller):
    '''A six-port mode-selective photonic lantern nuller

    Parameters
    ----------
    input_grid: Grid
        The grid on which the incoming wavefront is defined.
    mode_field_diameter: scalar
        The mode field diameter for the LP modes.
    pupil_diameter: scalar
        The pupil diameter
    focal_length: scalar
        The focal length for injection into the fiber.
    wavelength: scalar
        The wavelength of the simulation
    vortex_charge: integer
        (Optional) The charge of an optional pupil plane vortex mask.

    '''
    def __init__(self, input_grid, mode_field_diameter=1.31, pupil_diameter=1,focal_length=1, wavelength=1, vortex_charge=None):

        if vortex_charge is not None:
            phase_screen_gen = lambda grid: Field(vortex_charge * grid.as_('polar').theta, grid)
            phase_screen = phase_screen_gen(input_grid)
        else:
            phase_screen = np.zeros(input_grid.shape).ravel()

        super().__init__(input_grid,mode_field_diameter,pupil_diameter,focal_length,wavelength,phase_screen)

        lp_modes = make_LP_modes(self.focal_grid,1.5*np.pi,self.mode_field_diameter)
        normalized_modes = []
        for n in range(len(lp_modes)):
            normalized_modes.append(lp_modes[n]/np.sqrt(np.sum(np.abs(lp_modes[n])**2 * self.focal_grid.weights)))
        self.lp_modes = normalized_modes
        self.output_grid = CartesianGrid(RegularCoords([1, 1], [6, 1], np.zeros(2)))
        self.output_grid.weights = 1

    def forward(self, wavefront):
        '''Propagate a wavefront through the lantern nuller

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is expected to be
            in the pupil plane.

        Returns
        -------
        Field
            The coupling amplitudes through the nuller as complex scalars.
        '''
        wavelength = wavefront.wavelength

        post_phase_wf = self.phase_screen_optic.forward(wavefront)
        foc = self.prop.forward(post_phase_wf)

        output = np.zeros(len(self.lp_modes),dtype='complex128')

        for n in range(len(self.lp_modes)):
            output[n] = np.dot(foc.electric_field.conj() * self.focal_grid.weights, self.lp_modes[n].astype('complex128'))

        output = Field(output, self.output_grid)
        return output
        
    def backward(self, wavefront):
        '''Propagate backwards through the lantern nuller

        Parameters
        ----------
        wavefront : ndarray
            The wavefront to propagate. This wavefront is expected to be
            an ndarray of 6 complex scalars representing light emerging
            from the 6 ports of the lantern.

        Returns
        -------
        Wavefront
            The pupil plane wavefront.
        '''

        lant = Field(np.zeros(self.lp_modes[0].shape,dtype='complex128'),self.focal_grid)
        for n in range(len(self.lp_modes)):
            lant += self.lp_modes[n].astype('complex128')*wavefront[n]

        foc = Wavefront(lant,wavelength=self.wavelength)

        collimated = self.prop.backward(foc)
        output = self.phase_screen_optic.backward(collimated)

        return output