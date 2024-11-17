import numpy as np
from ..optics import OpticalElement, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field, CartesianGrid, RegularCoords
from ..mode_basis import ModeBasis

class PhotonicLantern(OpticalElement):
    '''A generic photonic lantern

    Parameters
    ----------
    input_grid: Grid
        The grid on which the incoming wavefront is defined.
    lantern_modes: ModeBasis
        The modes corresponding to the lantern ports.
    pupil_diameter: scalar
        The pupil diameter.
    focal_length: scalar
        The focal length for injection into the fiber.
    wavelength: scalar
        The wavelength of the simulation.
    '''

    def __init__(self, input_grid, lantern_modes, normalize_modes = True):
        self.input_grid = input_grid
        self.lantern_modes = lantern_modes
        self.num_modes = len(self.lantern_modes)
        self.focal_grid = self.lantern_modes.grid

        self.lantern_modes = [m / np.sqrt(np.sum(np.abs(m)**2 * self.focal_grid.weights)) for m in self.lantern_modes]
        self.lantern_modes = ModeBasis(self.lantern_modes)

        self.prop = FraunhoferPropagator(self.input_grid, self.focal_grid)
        self.output_grid = CartesianGrid(RegularCoords([1, 1], [self.num_modes, 1], np.zeros(2)))
        self.output_grid.weights = 1

        self.projection_matrix = self.lantern_modes.transformation_matrix

    def forward(self, wavefront):
        '''Forward propagate the light through the photonic lantern

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The complex amplitudes of each output port.
        '''

        foc = self.prop.forward(wavefront)
        output = self.projection_matrix.T.dot(foc.electric_field.conj() * self.focal_grid.weights)
        output = Field(output, self.output_grid)

        return Wavefront(output, wavefront.wavelength)

    def backward(self, wavefront):
        '''Backwards propagate the light through the photonic lantern.

        Parameters
        ----------
        wavefront : Wavefront
            The complex amplitudes for each of the input ports.

        Returns
        -------
        Wavefront
            The outgoing wavefront.
        '''
        res = self.projection_matrix.dot(wavefront.electric_field)
        return Wavefront(Field(res, self.focal_grid), wavefront.wavelength)