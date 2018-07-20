from .controller import Controller

from scipy.linalg import solve_discrete_are
from numpy.linalg import inv

class LinearGaussianRegulator(Controller):
	'''A Linear Gaussian Regulator controller.

	This class implements an infinite-horizon, discrete time LQR controller.

	Parameters
	----------
	A : ndarray
		The state transition matrix in the state space model.
	B : ndarray
		The input matrix in the state space model.
	Q : ndarray
		The cost matrix for the state.
	R : ndarray
		The cost matrix for the input.
	
	Attributes
	----------
	F : ndarray
		The computed control matrix.
	'''
	def __init__(self, A, B, Q, R):
		self.A = A
		self.B = B
		self.Q = Q
		self.R = R

	@property
	def A(self):
		return self._A
	
	@A.setter
	def A(self, A):
		self._A = A
		self._F = None
	
	@property
	def B(self):
		return self._B
	
	@B.setter
	def B(self, B):
		self._B = B
		self._F = None
	
	@property
	def Q(self):
		return self._Q
	
	@Q.setter
	def Q(self, Q):
		self._Q = Q
		self._F = None
	
	@property
	def R(self):
		return self._R
	
	@R.setter
	def R(self, R):
		self._R = R
		self._F = None
	
	@property
	def F(self):
		if self._F is None:
			P = solve_discrete_are(self.A, self.B, self.Q, self.R)
			self._F = inv(self.R + self.B.T.dot(P).dot(self.B)).dot(self.B.T.dot(P).dot(self.A))
		
		return self._F
	
	def submit_wavefront(self, t, wavefront, covariances=None, wfs_number=0):
		'''Submit a wavefront estimate to LQR.

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
		self.x = wavefront
	
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
		return -self.F.dot(self.x)
