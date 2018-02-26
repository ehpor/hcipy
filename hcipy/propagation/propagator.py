import numpy as np
from ..optics import OpticalElement, Wavefront

class MonochromaticPropagator(OpticalElement):
	def __init__(self, wavelength):
		self.wavelength = wavelength

def make_propagator(monochromatic_propagator):
	class Propagator(OpticalElement):
		def __init__(self, *args, **kwargs):
			self.wavelengths = []
			self.monochromatic_propagators = []
			self.monochromatic_args = args
			self.monochromatic_kwargs = kwargs
		
		def get_monochromatic_propagator(self, wavelength):
			if len(self.wavelengths) > 0:
				i = np.argmin(np.abs(wavelength - np.array(self.wavelengths)))
				wavelength_closest = self.wavelengths[i]

				delta_wavelength = np.abs(wavelength - wavelength_closest)
				if (delta_wavelength / wavelength) < 1e-6:
					return self.monochromatic_propagators[i]
			
			m = monochromatic_propagator(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)

			self.wavelengths.append(wavelength)
			self.monochromatic_propagators.append(m)

			if len(self.monochromatic_propagators) > 50:
				self.wavelengths.pop(0)
				self.monochromatic_propagators.pop(0)
				

			return m

		def __call__(self, wavefront):
			return self.forward(wavefront)
		
		def forward(self, wavefront):
			return self.get_monochromatic_propagator(wavefront.wavelength).forward(wavefront)
		
		def backward(self, wavefront):
			return self.get_monochromatic_propagator(wavefront.wavelength).backward(wavefront)
		
		def get_transformation_matrix_forward(self, wavelength=1):
			return self.get_monochromatic_propagator(wavelength).get_transformation_matrix_forward()

		def get_transformation_matrix_backward(self, wavelength=1):
			return self.get_monochromatic_propagator(wavelength).get_transformation_matrix_backward()

	return Propagator