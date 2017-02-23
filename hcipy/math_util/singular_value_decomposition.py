import numpy as np

class SVD(object):
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
		from ..mode_basis import ModeBasis
		
		return ModeBasis((m for m in self.U.conj().T))
	
	@property
	def right_singular_modes(self):
		from ..mode_basis import ModeBasis

		return ModeBasis((m for m in self.Vt.conj()))
	
	@property
	def singular_values(self):
		return self.S
	
	@property
	def U(self):
		return self.svd[0]
	
	@property
	def S(self):
		return self.svd[1]
	
	@property
	def Vt(self):
		return self.svd[2]
	
	def __getitem__(self, i):
		return self.svd[i]
	
	@property
	def svd(self):
		return self._svd
	
	@property
	def num_modes(self):
		return self._num_modes
	
	@property
	def M(self):
		return self._M

# Incremental SVD planned