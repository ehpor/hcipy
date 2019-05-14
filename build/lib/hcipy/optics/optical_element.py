import numpy as np
import inspect
import copy

class OpticalElement(object):
	def __call__(self, wavefront):
		return self.forward(wavefront)
	
	def forward(self, wavefront):
		raise NotImplementedError()
	
	def backward(self, wavefront):
		raise NotImplementedError()
	
	def get_transformation_matrix_forward(self, wavelength=1):
		raise NotImplementedError()
	
	def get_transformation_matrix_backward(self, wavelength=1):
		raise NotImplementedError()
	
	def get_instance(self, input_grid, wavelength):
		return self

class EmptyOpticalElement(OpticalElement):
	def forward(self, wavefront):
		return wavefront
	
	def backward(self, wavefront):
		return wavefront
	
	def get_transformation_matrix_forward(self, wavelength=1):
		return 1
	
	def get_transformation_matrix_backward(self, wavelength=1):
		return 1

def make_polychromatic(evaluated_arguments=None, num_in_cache=50):
	def decorator(optical_element):
		class PolychromaticOpticalElement(OpticalElement):
			def __init__(self, *args, **kwargs):
				self.wavelengths = []
				self.monochromatic_optical_elements = []
				self.monochromatic_args = args
				self.monochromatic_kwargs = kwargs

				if evaluated_arguments is not None:
					init = optical_element.__init__
					if hasattr(inspect, 'signature'):
						# Python 3
						monochromatic_arg_names = list(inspect.signature(init).parameters.keys())[1:]
					else:
						# Python 2
						monochromatic_arg_names = inspect.getargspec(init).args
					
					self.evaluate_arg = [m in evaluated_arguments for m in monochromatic_arg_names]

			def get_instance(self, input_grid, wavelength):
				if self.wavelengths:
					i = np.argmin(np.abs(wavelength - np.array(self.wavelengths)))
					wavelength_closest = self.wavelengths[i]

					delta_wavelength = np.abs(wavelength - wavelength_closest)
					if (delta_wavelength / wavelength) < 1e-6:
						return self.monochromatic_optical_elements[i]
				
				if evaluated_arguments is not None:
					args = list(self.monochromatic_args)
					kwargs = dict(self.monochromatic_kwargs)

					for i, (arg, ev) in enumerate(zip(args, self.evaluate_arg)):
						if ev and callable(arg):
							args[i] = arg(wavelength)
					
					for key, val in kwargs.items():
						if key in evaluated_arguments and callable(val):
							kwargs[key] = val(wavelength)
					
					elem = optical_element(*args, wavelength=wavelength, **kwargs)
				else:
					elem = optical_element(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)
				
				self.wavelengths.append(wavelength)
				self.monochromatic_optical_elements.append(elem)

				if len(self.wavelengths) > num_in_cache:
					self.wavelengths.pop(0)
					self.monochromatic_optical_elements.pop(0)
				
				return elem
			
			def forward(self, wavefront):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).forward(wavefront)
			
			def backward(self, wavefront):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).backward(wavefront)
			
			def get_transformation_matrix_forward(self, input_grid, wavelength):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_forward(input_grid, wavelength)
			
			def get_transformation_matrix_backward(self, input_grid, wavelength):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_backward(input_grid, wavelength)
		
		return PolychromaticOpticalElement
	return decorator

class OpticalSystem(OpticalElement):
	def __init__(self, optical_elements):
		self.optical_elements = optical_elements

	def forward(self, wavefront):
		wf = wavefront.copy()

		for optical_element in self.optical_elements:
			wf = optical_element.forward(wf)
		
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()

		for optical_element in reversed(self.optical_elements):
			wf = optical_element.backward(wf)
		
		return wf
	
	def get_transformation_matrix_forward(self, wavelength=1):
		matrix = 1

		for optical_element in self.optical_elements:
			matrix = np.dot(optical_element.get_transformation_matrix_forward(wavelength), matrix)
		
		return matrix
	
	def get_transformation_matrix_backward(self, wavelength=1):
		matrix = 1

		for optical_element in reversed(self.optical_elements):
			matrix = np.dot(optical_element.get_transformation_matrix_backward(wavelength), matrix)
		
		return matrix
	
	@property
	def optical_elements(self):
		return self._optical_elements

	@optical_elements.setter
	def optical_elements(self, optical_elements):
		self._optical_elements = list(optical_elements)
