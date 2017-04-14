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

class OpticalSystem(list, OpticalElement):

	def __init__(self, **kwargs):
		list.__init__(self, **kwargs)

	def forward(self, wavefront):
		temp = wavefront.copy()
		for oe in self.optical_elements:
			#pyplot.subplot(1,2,1)
			#imshow_field( temp.phase )
			print( temp.total_power )
			temp = oe.forward(temp)
			print( temp.total_power )
			#pyplot.subplot(1,2,2)
			#imshow_field( temp.phase )
			#pyplot.show()
		return temp
	
	def backward(self, wavefront):
		temp = wavefront.copy()
		for oe in self.optical_elements:
			temp = oe.backward(temp)
		return temp
	
	# String all optical elements together in a single matrix
	def get_transformation_matrix_forward(self, wavelength=1):
		raise NotImplementedError()
	
	def get_transformation_matrix_backward(self, wavelength=1):
		raise NotImplementedError()