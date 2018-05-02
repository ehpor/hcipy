class Controller(object):
	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		raise NotImplementedError()
	
	def get_actuators(self, t, dm_number=0):
		raise NotImplementedError()

class FilteredController(Controller):
	def __init__(self, control, filters):
		self.control = control
		if not hasattr(filters, '__iter__'):
			self.filters = [filters]
		else:
			self.filters = filters

	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		fltr = self.filters[wfs_number]
		filtered_wf, filtered_cov = fltr.filter(t, wavefront, covariances)

		self.control.submit_wavefront(t, filtered_wf, filtered_cov, wfs_number)

	def get_actuators(self, t, dm_number=0):
		return self.control.get_actuators(t, dm_number)