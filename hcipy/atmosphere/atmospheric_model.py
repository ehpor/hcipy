import numpy as np

class AtmosphericLayer(OpticalElement):
	def __init__(self, Cn_squared, velocity=0, height=0):
		self._velocity = None

		self.Cn_squared = Cn_squared
		self.velocity = velocity
		self.height = height
	
	def evolve_until(self, t):
		raise NotImplementedError()
	
	@property
	def Cn_squared(self):
		return self._Cn_squared
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		raise NotImplementedError()
	
	@property
	def fried_parameter(self):
		return self._r0
	
	@fried_parameter.setter
	def fried_parameter(self, r0):
		raise NotImplementedError()
	
	r0 = fried_parameter

	@property
	def velocity(self):
		return self._velocity

	@velocity.setter
	def velocity(self, velocity):
		if np.isscalar(velocity):
			if self._velocity is None:
				theta = np.random.rand() * 2*np.pi
				self._velocity = velocity * np.array([np.cos(theta), np.sin(theta)])
			else:
				self._velocity *= velocity / np.sqrt(np.dot(velocity, velocity))
		else:
			self._velocity = velocity

	def phase_for(self, wavelength):
		raise NotImplementedError()
	
	@property
	def output_grid(self):
		return self._output_grid

	@output_grid.setter
	def output_grid(self, output_grid):
		raise NotImplementedError()
	
	def forward(self, wf):
		wf = wf.copy()
		wf.electric_field *= np.exp(1j * self.phase_for(wf.wavelength))
		return wf
	
	def backward(self, wf):
		wf = wf.copy()
		wf.electric_field *= np.exp(-1j * self.phase_for(wf.wavelength))
	
class MultiLayerAtmosphere(OpticalElement):
	def __init__(self, layers, scintilation=False):
		self.layers = layers
		self.scintilation = scintilation

		self.calculate_propagators()

	def calculate_propagators(self):
		heights = [l.height for l in layers]
		self.layer_indices = np.argsort(-heights)

		sorted_heights = heights[self.layer_indices]
		delta_heights = sorted_heights[:-1] - sorted_heights[1:]

		self.propagators = [FresnelPropagator()]

	def forward(self, wf):
		pass
	
	def backward(self, wf):
		pass