import numpy as np
from ..propagation import FraunhoferPropagator

def generate_app_keller(wavefront, propagator, propagator_max, contrast, num_iterations):
	"""
	Modified Gerchberg-Saxton algorithm based on code by Christoph Keller.
	"""
	wf = wavefront.copy()

	for i in range(num_iterations):
		image = propagator.forward(wf)
		image_max = propagator_max.forward(wf).max()

		mask = image.intensity / image_max.intensity.max() > contrast

		if not np.any(mask):
			break
		
		delta_image = image.copy()
		delta_image.electric_field[~mask] = 0
		delta_image.electric_field[mask] *= -1#(1 - image_max.amplitude.max() * contrast[mask] / image.amplitude[mask])

		delta_aperture = propagator.backward(delta_image)
		wf.electric_field += delta_aperture.electric_field

		epsilon = 1e-6
		wf.electric_field[wf.amplitude < epsilon] = 0
		wf.electric_field[wf.amplitude >= epsilon] /= wf.amplitude[wf.amplitude >= epsilon]
	
	return wf

def generate_app_por(wavefront, propagator, propagator_max, contrast):
	raise NotImplementedError()