import numpy as np

from .multi_scale import MultiScaleCoronagraph
from ..field import Field

class FQPMCoronagraph(MultiScaleCoronagraph):
	def __init__(self, input_grid, lyot_stop=None, q=4, scaling_factor=10, window_size=128):
		phasor = lambda grid: Field(np.exp(1j * Field(np.abs(np.heaviside(grid.x, 0.5) - np.heaviside(grid.y, 0.5)) * np.pi, grid)), grid)
		super().__init__(input_grid, phasor, lyot_stop, q, scaling_factor, window_size)
