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
	def forward(self, wavefront, verbose=False):
		intermediate_wavefront = wavefront.copy()
		for optical_element in self:
			if verbose:
				print(intermediate_wavefront.total_power)
			intermediate_wavefront = optical_element.forward(intermediate_wavefront)
			if verbose:
				print(intermediate_wavefront.total_power)
		return intermediate_wavefront
	
	def backward(self, wavefront):
		intermediate_wavefront = wavefront.copy()
		for optical_element in self:
			intermediate_wavefront = optical_element.backward(intermediate_wavefront)
		return intermediate_wavefront
	
	# String all optical elements together in a single matrix
	def get_transformation_matrix_forward(self, wavelength=1):
		raise NotImplementedError()
	
	def get_transformation_matrix_backward(self, wavelength=1):
		raise NotImplementedError()