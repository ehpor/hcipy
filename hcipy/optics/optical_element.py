import numpy as np

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
