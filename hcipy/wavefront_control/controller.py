class Controller(object):
	def submit_wavefront(self, t, wavefront, covariances=None):
		raise NotImplementedError()
	
	def get_actuators(self, t):
		raise NotImplementedError()

class FilteredController(Controller):
	def __init__(self, control, reconstructor):
		self.control = control
		self.reconstructor = reconstructor

	def submit_wavefront(self, t, wavefront, covariances=None):
		filtered_wf, filtered_cov = self.reconstructor.reconstruct(t, wavefront, covariances)
		self.control.submit_wavefront(t, filtered_wf, filtered_cov)

	def get_actuators(self, t):
		return self.control.get_actuators(t)