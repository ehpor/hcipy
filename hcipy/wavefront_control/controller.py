class Controller(object):
	'''A controller in an adaptive optics system.
	'''
	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		'''Submit a wavefront estimate to the controller.

		Parameters
		----------
		t : scalar
			Time at which the estimate was taken.
		wavefront : Field
			The estimate of the wavefront. This can be slopes, mode coefficients, etc...
		covariances : ndarray or scalar
			The structure of the covariance matrix. If this is a scalar, elements of wavefront
			are assumed independent and distributed with a variance `covariances`. If this is
			a vector, all elements are independent and have a variance `covariances`. If this is
			a matrix, a full covariance matrix is assumed.
		wfs_number : int
			The index of the wavefront sensor. This is meant for support of multiple
			wavefront sensors.
		'''
		raise NotImplementedError()
	
	def get_actuators(self, t, dm_number=0):
		'''Get the actuator positions at time `t` for DM number `dm_number`.

		Parameters
		----------
		t : scalar
			The time at which to get the requested actuator positions.
		dm_number : int
			The index of the deformable mirror. This is meant for support for multiple
			deformable mirrors.
		'''
		raise NotImplementedError()

class ObserverController(Controller):
	'''A controller based on the separation principle of control theory.

	Parameters
	----------
	controller : Controller
		The controller to use.
	observers : list of Observers or Observer
		The observer for each of the wavefront sensors. If only one observer is given,
		a single wavefront sensor is assumed.
	'''
	def __init__(self, controller, observers):
		self.controller = controller
		if not hasattr(observers, '__iter__'):
			self.observers = [observers]
		else:
			self.observers = observers

	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		'''Submit a wavefront estimate to the controller.

		Parameters
		----------
		t : scalar
			Time at which the estimate was taken.
		wavefront : Field
			The estimate of the wavefront. This can be slopes, mode coefficients, etc...
		covariances : ndarray or scalar
			The structure of the covariance matrix. If this is a scalar, elements of wavefront
			are assumed independent and distributed with a variance `covariances`. If this is
			a vector, all elements are independent and have a variance `covariances`. If this is
			a matrix, a full covariance matrix is assumed.
		wfs_number : int
			The index of the wavefront sensor. This is meant for support of multiple
			wavefront sensors.
		'''
		observer = self.observers[wfs_number]
		filtered_wf, filtered_cov = observer.estimate(t, wavefront, covariances)

		self.control.submit_wavefront(t, filtered_wf, filtered_cov, wfs_number)

	def get_actuators(self, t, dm_number=0):
		'''Get the actuator positions at time `t` for DM number `dm_number`.

		Parameters
		----------
		t : scalar
			The time at which to get the requested actuator positions.
		dm_number : int
			The index of the deformable mirror. This is meant for support for multiple
			deformable mirrors.
		'''
		return self.control.get_actuators(t, dm_number)