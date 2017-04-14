from ..plotting import *
from matplotlib import pyplot

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

class OpticalSystem(list):

	def forward(self, wavefront):
		temp = wavefront.copy()
		for i, optical_element in enumerate(self):
			temp = optical_element.forward(temp)
		return temp
	
	def backward(self, wavefront):
		temp = wavefront.copy()
		for optical_element in self:
			temp = optical_element.backward(temp)
		return temp
	
	# String all optical elements together in a single matrix
	def get_transformation_matrix_forward(self, wavelength=1):
		raise NotImplementedError()
	
	def get_transformation_matrix_backward(self, wavelength=1):
		raise NotImplementedError()