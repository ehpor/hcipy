import numpy as np
from .optical_element import OpticalElement
from ..field import CartesianGrid, UnstructuredCoords

class PeriodicOpticalElement(OpticalElement):
    def __init__(self, input_grid, pitch, apodization, orientation=0, even_grid=False):
        '''An even asphere micro-lens array.

        Parameters
        ----------
        input_grid : Grid
            The grid on which the periodic optical element is evaluated.
        pitch : scalar
            The pitch of the periodic optical element.
        apodization : Apodizer
            The apodizer that will be evaluated on the periodic grid.
        orientation : scalar
            The orientation of the periodic optical element.
        even_grid : bool
            This determines whether zero is in between two elements or if it is the center of an element.
        '''
        self.input_grid = input_grid.copy()
        self.input_grid = self.input_grid.rotated(orientation)

        if even_grid:
            xf = (np.fmod(abs(self.input_grid.x), pitch) - pitch / 2) * np.sign(self.input_grid.x)
            yf = (np.fmod(abs(self.input_grid.y), pitch) - pitch / 2) * np.sign(self.input_grid.y)
        else:
            xf = (np.fmod(abs(self.input_grid.x) + pitch / 2, pitch) - pitch / 2) * np.sign(self.input_grid.x)
            yf = (np.fmod(abs(self.input_grid.y) + pitch / 2, pitch) - pitch / 2) * np.sign(self.input_grid.y)

        periodic_grid = CartesianGrid(UnstructuredCoords((xf, yf)))
        self.apodization = apodization(periodic_grid)

    def forward(self, wavefront):
        wf = wavefront.copy()
        return self.apodization.forward(wf)

    def backward(self, wavefront):
        wf = wavefront.copy()
        return self.apodization.backward(wf)
