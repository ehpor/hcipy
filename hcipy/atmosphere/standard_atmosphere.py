from .infinite_atmospheric_layer import InfiniteAtmosphericLayer
from .atmospheric_model import MultiLayerAtmosphere

import numpy as np

def make_standard_atmospheric_layers(input_grid, L0=10):
	heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 10, 10, 10, 10])
	Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350]) * 1e-12

	layers = []
	for h, v, cn in zip(heights, velocities, Cn_squared):
		layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))
	
	return layers