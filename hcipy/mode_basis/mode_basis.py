import numpy as np

class ModeBasis(object):
	def __init__(self, basis_vectors):
		if hasattr(basis_vectors, 'ndim'):
			self._transformation_matrix = basis_vectors
		else:
			self._transformation_matrix = np.column_stack(basis_vectors))

	def make_linear_combination(self, coeff):
		return np.dot(self.matrix, coeff)

	@property
	def transformation_matrix(self):
		return self._transformation_matrix
	
	def __getitem__(self, i):
		if i >= len(self):
			raise IndexError()
		return self.transformation_matrix[:,i]
	
	def __setitem__(self, i, value):
		if i >= len(self):
			raise IndexError()
		return self.transformation_matrix[:,i] = value

	def __len__(self):
		return self.matrix.shape[1]
	
	@property
	def orthogonalized(self):
		q, r = np.linalg.qr(self.transformation_matrix)
		return ModeBasis(q)
 