import numpy as np

from .multi_scale import MultiScaleCoronagraph
from ..field import Field

class FQPMCoronagraph(MultiScaleCoronagraph):
    def __init__(self, input_grid, lyot_stop=None, q=4, scaling_factor=4, window_size=128):
        phase = lambda grid: Field(np.exp(1j * np.pi * np.sign(grid.x * grid.y)), grid)
        super().__init__(input_grid, phase, lyot_stop, q, scaling_factor, window_size)
