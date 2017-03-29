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

class Detector(object):
	def integrate(self, wavefront, dt, weight=1):
		raise NotImplementedError()
	
	def read_out(self):
		raise NotImplementedError()
	
	def __call__(self, wavefront, dt=1, weight=1):
		self.integrate(wavefront, dt, weight)
		return self.read_out()