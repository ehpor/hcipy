from .atmospheric_model import Cn_squared_from_fried_parameter
from .infinite_atmospheric_layer import InfiniteAtmosphericLayer

import numpy as np

def make_standard_atmospheric_layers(input_grid, L0=10):
	heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 10, 10, 10, 10])
	Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350]) * 1e-12

	layers = []
	for h, v, cn in zip(heights, velocities, Cn_squared):
		layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

	return layers

def make_lco_atmospheric_layers(input_grid, r0=0.16, L0=25):
	heights = np.array([250, 500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 20, 20, 25, 30, 25])
	
	integrated_cn_squared = Cn_squared_from_fried_parameter(r0, wavelength=500e-9)
	Cn_squared = np.array([0.42, 0.03, 0.06, 0.16, 0.11, 0.10, 0.12]) * integrated_cn_squared
			
	layers = []
	for h, v, cn in zip(heights, velocities, Cn_squared):
		layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))

	return layers