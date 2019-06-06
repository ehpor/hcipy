import numpy as np
import scipy.sparse

from ..field import Field
'''
class ModeBasis(list):
	@property
	def transformation_matrix(self):
		return np.stack(self, axis=-1)
	
	# TODO: currently only works on scalar fields
	@property
	def orthogonalized(self):
		q, r = np.linalg.qr(self.transformation_matrix)
		if hasattr(self[0], 'grid'):
			from ..field import Field
			grid = self[0].grid
			return ModeBasis((Field(q[:,i], grid) for i in range(q.shape[1])))
		else:
			return ModeBasis((q[:,i] for i in range(q.shape[1])))
'''
class ModeBasis(object):
	'''A list of modes.

	Parameters
	----------
	transformation_matrix : array_like or list of array_like
		The transformation matrix of the mode basis or a list of modes.
	grid : Grid or None
		The grid on which the modes are defined.
	'''
	def __init__(self, transformation_matrix, grid=None):
		if scipy.sparse.issparse(transformation_matrix):
			self._transformation_matrix = transformation_matrix.to_csc()
		elif scipy.sparse.issparse(transformation_matrix[0]):
			self._transformation_matrix = scipy.sparse.bmat(transformation_matrix, format='csc')
		elif isinstance(transformation_matrix, (list, tuple)):
			self._transformation_matrix = np.stack(transformation_matrix, axis=-1)
		else:
			self._transformation_matrix = transformation_matrix

		if grid is not None:
			self.grid = grid
		elif hasattr(transformation_matrix[0], 'grid'):
			self.grid = transformation_matrix[0].grid
		else:
			self.grid = None
	
	@property
	def is_sparse(self):
		'''If the mode basis is sparse.
		'''
		return scipy.sparse.issparse(self._transformation_matrix)
	
	@property
	def is_dense(self):
		'''If the mode basis is dense.
		'''
		return not self.is_sparse
	
	@property
	def transformation_matrix(self):
		'''The transformation matrix of this mode basis.
		'''
		return self._transformation_matrix
	
	def coefficients_for(self, b, dampening_factor=0):
		'''Calculate the coefficients on this mode basis in a least squares fashion.

		The vector `b` is projection onto the mode basis in a least squares fashion. This
		means that the const function

		.. math:: J(c) = |b - A x|^2_2 + |\lambda x|^2_2

		is minimized, where :math:`x` are the coefficients, and :math:`\lambda` is the 
		dampening factor.

		If this projection needs to be done repeatedly, you may be better off calculating 
		the inverse of the transformation matrix directly and left-multiplying that with 
		your vector, rather than using a least squares estimation every time.

		Parameters
		----------
		b : array_like or Field
			The vector for which to calculate the coefficients.
		dampening_factor : scalar
			The Tikhonov dampening factor used for the least squares procedure.
		
		Returns
		-------
		array_like
			The coefficients that correspond to the vector `b`.
		'''
		if self.is_sparse or dampening_factor != 0:
			x, istop, itn, normr, norma, conda, normx = scipy.sparse.linalg.lsmr(self._transformation_matrix, b, damp=dampening_factor)
			return x
		else:
			x, residuals, rank, s = np.linalg.lstsq(self._transformation_matrix, b)
			return x
	
	def linear_combination(self, coefficients):
		'''Calculate a linear combination using this mode basis.

		Parameters
		----------
		coefficients : array_like or list
			The coefficients of the linear combinations.
		
		Returns
		-------
		array_like or Field
			The calculated linear combination.
		'''
		if self.grid is None:
			return self._transformation_matrix.dot(coefficients)
		else:
			return Field(self._transformation_matrix.dot(coefficients), self.grid)
	
	@property
	def orthogonalized(self):
		'''Get an orthogonalized version of this ModeBasis.

		The resulting ModeBasis spans the same vector space, but each mode is orthogonal to 
		all others. In general the resulting `ModeBasis` is dense, so no distinction is made
		between sparse and dense mode bases in this function. This function will always return
		a dense mode basis.

		Returns
		-------
		ModeBasis
			A mode basis with orthogonalized modes.
		'''
		q, r = np.linalg.qr(self.transformation_matrix)
		return ModeBasis(q, self.grid)

	def __getitem__(self, item):
		'''Get the `item`-th mode in the `ModeBasis`.



		Parameters
		----------
		item : int or slice or array_like
			The index/indices of the mode(s).
		
		Returns
		-------
		Field or array_like or ModeBasis
			The `item`-th mode in the `ModeBasis`.
		'''
		T = self._transformation_matrix[..., item]
		
		if T.ndim == self._transformation_matrix.ndim:
			# We are returning multiple modes; put these in a ModeBasis.
			return ModeBasis(T, self.grid)
		else:
			# We are returning a single mode; return just this.
			if self.grid is None:
				return self._transformation_matrix[..., item]
			else:
				return Field(self._transformation_matrix[..., item], self.grid)
	
	def __len__(self):
		'''The number of modes in the `ModeBasis`.

		Returns
		-------
		int
			The number of modes in the `ModeBasis`.
		'''
		return self.transformation_matrix.shape[0]
	
	def append(self, mode):
		'''Append `mode` to this mode basis.

		Parameters
		----------
		mode : array_like or Field
			The mode to add to the ModeBasis
		'''
		if self.is_sparse:
			self._transformation_matrix = scipy.sparse.vstack((self._transformation_matrix, mode), 'csr')
		else:
			self._transformation_matrix = np.vstack((self._transformation_matrix, mode))
	
	def extend(self, modes):
		'''Extend the mode basis with `modes`.

		Parameters
		----------
		modes : list or array_like or ModeBasis
			The modes to add to the ModeBasis.
		'''
		if isinstance(modes, ModeBasis):
			modes = modes.transformation_matrix
		
		if self.is_sparse:
			self._transformation_matrix = scipy.sparse.vstack((self._transformation_matrix, modes), 'csr')
		else:
			self._transformation_matrix = np.vstack((self._transformation_matrix, modes))
	
	def __add__(self, mode_basis):
		'''Merge two mode bases into one.

		Parameters
		----------
		mode_basis : ModeBasis
			The ModeBasis to add.
		
		Returns
		-------
		ModeBasis
			The newly created `ModeBasis`.
		'''
		if self.is_sparse or mode_basis.is_sparse:
			transformation_matrix = scipy.sparse.vstack((self._transformation_matrix, mode_basis.transformation_matrix), 'csr')
		else:
			transformation_matrix = np.vstack((self._transformation_matrix, mode_basis.transformation_matrix))
		return ModeBasis(transformation_matrix, self.grid)
