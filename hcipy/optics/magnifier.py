import numpy as np

from .optical_element import AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from .wavefront import Wavefront
from ..field import Field

class Magnifier(AgnosticOpticalElement):
	def __init__(self, magnification):
		''' An ideal magnifier.

		This class magnifies a wavefront in an energy conserving manner.

		Parameters
		----------
		magnification : scalar or function of wavelength
			The magnification of the system.
		'''
		self._magnification = magnification

		AgnosticOpticalElement.__init__(self, False, True)
	
	def make_instance(self, instance_data, input_grid, output_grid, wavelength):
		instance_data.magnification = self.evaluate_parameter(self._magnification, input_grid, output_grid, wavelength)
	
	@property
	def magnification(self):
		return self._magnification
	
	@magnification.setter
	def magnification(self, magnification):
		self._magnification = magnification

		self.clear_cache()
	
	def get_input_grid(self, output_grid, wavelength):
		magnification = self.evaluate_parameter(self._magnification, None, None, wavelength)

		return output_grid.scaled(1.0 / magnification)
	
	def get_output_grid(self, input_grid, wavelength):
		magnification = self.evaluate_parameter(self._magnification, input_grid, None, wavelength)

		return input_grid.scaled(magnification)
	
	@make_agnostic_forward
	def forward(self, instance_data, wavefront):
		wf = wavefront.copy()

		wf.electric_field.grid = wf.electric_field.grid.scaled(instance_data.magnification)
		wf.electric_field /= instance_data.magnification

		return wf
	
	@make_agnostic_backward
	def backward(self, instance_data, wavefront):
		wf = wavefront.copy()

		wf.electric_field.grid = wf.electric_field.grid.scaled(1.0 / instance_data.magnification)
		wf.electric_field *= instance_data.magnification

		return wf
