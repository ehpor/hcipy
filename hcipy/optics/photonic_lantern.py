import numpy as np

from ..optics import OpticalElement, Wavefront
from ..field import Field, CartesianGrid, RegularCoords
from ..mode_basis import ModeBasis

class PhotonicLantern(OpticalElement):
    '''A generic photonic lantern

    Parameters
    ----------
    lantern_modes : ModeBasis
        The modes corresponding to the lantern ports.
    wavelength : scalar
        The wavelength of the simulation.
    normalize_modes : bool
        Normalize the modes to unit power or not. Default is True.
    '''
    def __init__(self, lantern_modes, normalize_modes=True):
        self.lantern_modes = lantern_modes
        self.num_modes = len(self.lantern_modes)
        self.input_grid = self.lantern_modes.grid

        if normalize_modes:
            self.lantern_modes = [m / np.sqrt(np.sum(np.abs(m)**2 * self.input_grid.weights)) for m in self.lantern_modes]
            self.lantern_modes = ModeBasis(self.lantern_modes)

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
        output = self.projection_matrix.T.dot(wavefront.electric_field.conj() * self.input_grid.weights)
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

        return Wavefront(Field(res, self.input_grid), wavefront.wavelength)
