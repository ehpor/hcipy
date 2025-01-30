# import numpy as np
from ..math import numpy as np

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
    mode_field_diameter : scalar
        (Optional) The mode field diameter of the lantern modes.
    vortex_charge : integer
        (Optional) The charge of an optional pupil plane vortex mask.
    '''
    def __init__(self, input_grid, focal_grid, mode_field_diameter=1.31, vortex_charge=None):
        lp_modes = make_lp_modes(focal_grid, 1.5 * np.pi, mode_field_diameter)
        photonic_lantern = PhotonicLantern(lp_modes)

        if vortex_charge is not None:
            phase_screen = Field(vortex_charge * input_grid.as_('polar').theta, input_grid)
            phase_apodizer = PhaseApodizer(phase_screen)
        else:
            phase_apodizer = None

        super().__init__(input_grid, photonic_lantern, phase_apodizer)
