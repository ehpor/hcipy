class Controller(object):
	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		raise NotImplementedError()
	
	def get_actuators(self, t, dm_number=0):
		raise NotImplementedError()

class FilteredController(Controller):
	def __init__(self, control, reconstructors):
		self.control = control
		if not hasattr(reconstructors, '__iter__'):
			self.reconstructors = [reconstructors]
		else:
			self.reconstructors = reconstructors

	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		recon = self.reconstructors[wfs_number]
		filtered_wf, filtered_cov = recon.reconstruct(t, wavefront, covariances)

		self.control.submit_wavefront(t, filtered_wf, filtered_cov, wfs_number)

	def get_actuators(self, t, dm_number=0):
		return self.control.get_actuators(t, dm_number)