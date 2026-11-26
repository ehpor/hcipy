import numpy as np
import warnings

from ..optics import OpticalElement, PhaseApodizer, PhotonicLantern
from ..propagation import FraunhoferPropagator
from ..field import Field, CartesianGrid, RegularCoords
from ..mode_basis import make_lp_modes

class FiberNuller(OpticalElement):
    '''A generic fiber nuller

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    fiber : OpticalElement
        The fiber the light is injected into.
    apodizer : OpticalElement
        The apodizer, assumed to be in the pupil plane.
    '''
    def __init__(self, input_grid, fiber, apodizer=None):
        self.input_grid = input_grid
        self.fiber = fiber
        self.focal_grid = self.fiber.input_grid
        self.apodizer = apodizer

        self.prop = FraunhoferPropagator(self.input_grid, self.focal_grid)

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
        Wavefront
            The coupling amplitude through the fiber nuller.
        '''
        if self.apodizer is not None:
            wavefront = self.apodizer.forward(wavefront)

        foc = self.prop.forward(wavefront)
        output = self.fiber.forward(foc)

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
        foc = self.fiber.backward(wavefront)
        output = self.prop.backward(foc)

        if self.apodizer is not None:
            output = self.apodizer.backward(output)

        return output

class VortexFiberNuller(FiberNuller):
    '''A vortex fiber nuller

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    fiber : OpticalElement
        The fiber the light is injected into.
    vortex_charge : integer
        The vortex charge.
    '''
    def __init__(self, input_grid, fiber, vortex_charge=1):
        phase_screen = Field(vortex_charge * input_grid.as_('polar').theta, input_grid)
        phase_apodizer = PhaseApodizer(phase_screen)

        super().__init__(input_grid, fiber, phase_apodizer)

class PhotonicLanternNuller(FiberNuller):
    '''A 6 port mode-selective photonic lantern nuller

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    focal_grid : Grid
        The focal grid where light is injected into the lantern.
    core_radius : scalar
        The core radius of the step-index fiber used to model the lantern modes.
        Note that this is a radius, not a mode-field diameter. This argument is required.
    vortex_charge : integer
        (Optional) The charge of an optional pupil plane vortex mask.
    mode_field_diameter : scalar
        Deprecated alias for ``core_radius``. This argument will be removed in a
        future version.
    '''
    def __init__(self, input_grid, focal_grid, core_radius=None, vortex_charge=None, mode_field_diameter=None):
        if mode_field_diameter is not None:
            if core_radius is not None:
                raise TypeError('Specify either core_radius or mode_field_diameter, not both.')

            warnings.warn(
                'The mode_field_diameter argument is deprecated. Use core_radius instead. '
                'The supplied value is still interpreted as the core radius for backward compatibility.',
                DeprecationWarning,
                stacklevel=2
            )
            core_radius = mode_field_diameter

        if core_radius is None:
            raise TypeError('core_radius is required. The mode_field_diameter argument is deprecated; use core_radius instead.')

        self.core_radius = core_radius
        lp_modes = make_lp_modes(focal_grid, 1.5 * np.pi, core_radius)
        photonic_lantern = PhotonicLantern(lp_modes)

        if vortex_charge is not None:
            phase_screen = Field(vortex_charge * input_grid.as_('polar').theta, input_grid)
            phase_apodizer = PhaseApodizer(phase_screen)
        else:
            phase_apodizer = None

        super().__init__(input_grid, photonic_lantern, phase_apodizer)
