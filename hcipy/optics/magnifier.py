import numpy as np
from .optical_element import OpticalElement, make_agnostic_optical_element
from .wavefront import Wavefront
from ..field import Field

@make_agnostic_optical_element([], ['magnification'])
class Magnifier(OpticalElement):
	'''A monochromatic magnifier for electric fields.
	
	This magnifies the wavefront with a certain magnification factor.
	It does not take into acount propagation effects.
	
	Parameters
	----------
	magnification : scalar
		The magnification we want to apply to the grid of the wavefront.
	wavelength : scalar
		The wavelength at which the magnification is defined.
	'''
	def __init__(self, magnification, wavelength=1):
		self.magnification = magnification
		self.wavelength = wavelength
	
	def forward(self, wavefront):

		new_grid = wavefront.electric_field.grid.scaled(self.magnification)

		wf = Wavefront(Field(wavefront.electric_field.copy(), new_grid), wavefront.wavelength)
		wf.total_power = wavefront.total_power

		return wf
	
	def backward(self, wavefront):

		new_grid = wavefront.electric_field.grid.scaled(1/self.magnification)

		wf = Wavefront(Field(wavefront.electric_field.copy(), new_grid), wavefront.wavelength)
		wf.total_power = wavefront.total_power

		return wf