import numpy as np

class SVD(object):
	'''The Singular Value Decomposition for the provided matrix.

	This class wraps two versions of the SVD in numpy and scipy, and provides
	easy access to singular modes (as mode bases) and allows for calculation
	of the SVD for a limited number of modes.

	Parameters
	----------
	M : np.ndarray
		The matrix on which to perform the SVD.
	num_modes : int or None
		The number of singular values and modes to calculate. If this is None,
		all modes will be computed.
	'''
	def __init__(self, M, num_modes=None):
		self._M = M
		self._num_modes = num_modes

		if self.num_modes is None:
			from numpy.linalg import svd
			self._svd = svd(M, full_matrices=False)
		else:
			from scipy.sparse.linalg import svds
			self._svd = svds(M, int(self.num_modes))
	
	@property
	def left_singular_modes(self):
		'''The left singular modes of the matrix, as a ModeBasis.
		'''
		from ..mode_basis import ModeBasis
		
		return ModeBasis((m for m in self.U.conj().T))
	
	@property
	def right_singular_modes(self):
		'''The right singular modes of the matrix, as a ModeBasis.
		'''
		from ..mode_basis import ModeBasis

		return ModeBasis((m for m in self.Vt.conj()))
	
	@property
	def singular_values(self):
		'''The singular values of the matrix.
		'''
		return self.S
	
	@property
	def U(self):
		'''The U matrix of the SVD.
		'''
		return self.svd[0]
	
	@property
	def S(self):
		'''The singular values of the matrix.
		'''
		return self.svd[1]
	
	@property
	def Vt(self):
		'''The V^T matrix of the SVD.
		'''
		return self.svd[2]
	
	def __getitem__(self, i):
		'''The raw U, S, and V^T matrices of the SVD by index.
		'''
		return self.svd[i]
	
	@property
	def svd(self):
		'''The raw U, S, and V^T matrices of the SVD as a tuple.
		'''
		return self._svd
	
	@property
	def num_modes(self):
		'''The number of singular modes that were calculated in this SVD.
		'''
		return self._num_modes
	
	@property
	def M(self):
		'''The matrix for which the SVD was calculated.
		'''
		return self._M
