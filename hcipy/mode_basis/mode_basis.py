import numpy as np
import scipy.sparse

from ..field import Field

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
			sparse = True
			is_list = False
		elif scipy.sparse.issparse(transformation_matrix[0]):
			sparse = np.all([transformation_matrix[i].shape[0] == 1 for i in range(len(transformation_matrix))])
			is_list = True
		else:
			sparse = False
			is_list = isinstance(transformation_matrix, (list, tuple))
		
		if sparse:
			if is_list:
				self._modes = scipy.sparse.vstack(transformation_matrix, format='csr')
				self._transformation_matrix = self._modes.T.tocsc()
			else:
				self._modes = transformation_matrix.tocsr()
				self._transformation_matrix = transformation_matrix.tocsc()
		else:
			if is_list:
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

	def to_sparse(self, copy=False):
		'''Convert the mode basis to a sparse mode basis.

		Parameters
		----------
		copy : boolean
			Whether to force a copy or not. A copy is always made if 
			the current ModeBasis is not sparse.
		
		Returns
		-------
		ModeBasis
			The sparsified ModeBasis.
		
		Raises
		------
			TypeError
				If this ModeBasis cannot be sparsified.
		'''
		if self.is_sparse:
			if copy:
				return ModeBasis(self._transformation_matrix.copy(), self.grid)
			else:
				return self
		else:
			if self._transformation_matrix.ndim != 2:
				raise TypeError('Cannot sparsify a mode basis of tensor fields')
			
			T = scipy.sparse.csc_matrix(self._transformation_matrix)
			T.eliminate_zeros()
			return ModeBasis(T, self.grid)

	def to_dense(self, copy=False):
		'''Convert the mode basis to a dense mode basis.

		Parameters
		----------
		copy : boolean
			Whether to force a copy or not. A copy is always made if 
			the current ModeBasis is not dense.
		
		Returns
		-------
		ModeBasis
			The densified ModeBasis.
		'''
		if self.is_dense:
			if copy:
				return ModeBasis(self._transformation_matrix.copy(), self.grid)
			else:
				return self
		else:
			T = self._transformation_matrix.todense()
			return ModeBasis(T, self.grid)

	@property
	def transformation_matrix(self):
		'''The transformation matrix of this mode basis.
		'''
		return self._transformation_matrix
	
	@transformation_matrix.setter
	def transformation_matrix(self, transformation_matrix):
		self._transformation_matrix = transformation_matrix

	def coefficients_for(self, b, dampening_factor=0):
		r'''Calculate the coefficients on this mode basis in a least squares fashion.

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
		y = self._transformation_matrix.dot(coefficients)

		if self.grid is None:
			return y
		else:
			return Field(y, self.grid)

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
		
		Raises
		------
		NotImplementedError
			If the mode basis is a mode basis containing non-scalar fields.
		'''
		if self._transformation_matrix.ndim != 2:
			raise NotImplementedError('The mode basis contains non-scalar fields; orthogonalization is not implemented for these.')

		q, r = np.linalg.qr(self._transformation_matrix)
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
		
		return_mode_basis = False
		if self.is_sparse:
			if T.shape[-1] != 1:
				return_mode_basis = True
		if self.is_dense:
			if T.ndim == self._transformation_matrix.ndim:
				return_mode_basis = True
		
		if return_mode_basis:
			# We are returning multiple modes; put these in a ModeBasis.
			return ModeBasis(T, self.grid)
		else:
			# We are returning a single mode; return just this.
			if self.is_sparse:
				T = T.toarray()[...,0]
			
			if self.grid is None:
				return T
			else:
				return Field(T, self.grid)

	def __len__(self):
		'''The number of modes in the `ModeBasis`.

		Returns
		-------
		int
			The number of modes in the `ModeBasis`.
		'''
		return self.transformation_matrix.shape[-1]

	def append(self, mode):
		'''Append `mode` to this mode basis.

		Parameters
		----------
		mode : array_like or Field
			The mode to add to the ModeBasis
		'''
		if self.is_sparse:
			self._transformation_matrix = scipy.sparse.hstack((self._transformation_matrix, mode), 'csc')
		else:
			self._transformation_matrix = np.concatenate((self._transformation_matrix, [mode]), axis=-1)

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
			self._transformation_matrix = scipy.sparse.hstack((self._transformation_matrix, modes), 'csc')
		else:
			# TODO: worry about modes being in a list instead of transformation matrix or mode basis
			self._transformation_matrix = np.concatenate((self._transformation_matrix, modes), axis=-1)

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
			transformation_matrix = scipy.sparse.hstack((self._transformation_matrix, mode_basis.transformation_matrix), 'csc')
		else:
			transformation_matrix = np.concatenate((self._transformation_matrix, mode_basis.transformation_matrix), axis=-1)
		return ModeBasis(transformation_matrix, self.grid)
