import numpy as np

from .multi_scale import MultiScaleCoronagraph
from ..field import Field

class FQPMCoronagraph(MultiScaleCoronagraph):
    def __init__(self, input_grid, lyot_stop=None, q=1024, scaling_factor=4, window_size=32):
        complex_mask = lambda grid: Field(np.sign(grid.x) * np.sign(grid.y), grid).astype('complex')
        super().__init__(input_grid, complex_mask, lyot_stop, q, scaling_factor, window_size)
