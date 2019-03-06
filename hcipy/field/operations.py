from .field import Field
import numpy as np

def field_inverse_tikhonov(f, rcond=1e-15):
	'''Invert a tensor field of order 2 using Tikhonov regularization.

	Parameters
	----------
	f : `Field`
		The tensor field for which to calculate the inverses. The tensor order
		of this Field has to be 2.
	rcond : scalar
		The relative regularization parameter to use for the inversions.
	
	Returns
	-------
	Field
		The resulting Field with tensor order 2.
	
	Raises
	------
	ValueError
		If the tensor order of field `f` is not 2.
	'''
	from ..math_util import inverse_tikhonov

	if f.tensor_order != 2:
		raise ValueError("Field must be a tensor field of order 2 to be able to calculate inverses.")

	res = []
	for i in range(f.grid.size):
		res.append(inverse_tikhonov(f[...,i], rcond))
	return Field(np.moveaxis(res, 0, -1), f.grid)

def field_svd(f, full_matrices=True, compute_uv=True):
	'''Calculate the singular value decomposition for a tensor field of order 2.

	Parameters
	----------
	f : `Field`
		The tensor field for which to calculate the singular value decompositions.
	full_matrices : boolean
		If True, matrices in U and Vh will have shapes (M,M) and (N,N) respectively.
		Otherwise their shapes are (M,K), (K,N) respectively, where K=min(M,N).
	compute_uv : boolean
		Whether to compute matrices U and Vh in addition to the singular values.
	
	Returns
	-------
	U : `Field`
		The unitary matrices. Only returned if `compute_uv` is True.
	S : `Field`
		The singular values, sorted in descending order.
	Vh : `Field`
		The unitary matrices. Only returned if `compute_uv` is True.
	'''
	
	res = np.linalg.svd(np.moveaxis(f, -1, 0), full_matrices, compute_uv)

	if compute_uv:
		U, S, Vh = res
		U = Field(np.moveaxis(U, 0, -1), f.grid)
		Vh = Field(np.moveaxis(Vh, 0, -1), f.grid)
	else:
		S = res
	
	S = Field(np.moveaxis(S, 0, -1), f.grid)
	
	if compute_uv:
		return U, S, Vh
	else:
		return S

def field_conjugate_transpose(a):
	'''Performs the conjugate transpose of a rank 2 tensor field.

	Parameters
	----------
	a : Field
		The field to conjugate transpose

	Returns
	-------
	Field
		The conjugate transposed field
	'''
	
	if a.tensor_order != 2:
		raise ValueError('Need a tensor field of rank 2.')

	return Field(np.swapaxes(a.conj(),0,1), a.grid)

def field_transpose(a):
	'''Performs the transpose of a rank 2 tensor field.

	Parameters
	----------
	a : Field
		The field to transpose

	Returns
	-------
	Field
		The transposed field
	'''
	if a.tensor_order != 2:
		raise ValueError('Need a tensor field of rank 2.')

	return Field(np.swapaxes(a,0,1), a.grid)

def field_determinant(a):
	'''Calculates the determinant of a tensor field.

	Parameters
	----------
	a : Field
		The field for which the determinant needs to be calculated

	Returns
	-------
	Field
		The field that contains the determinant on every spatial position
	'''
	if a.tensor_order == 1:
		raise ValueError('Only tensor fields of order 2 or higher have a determinant.')
		
	if a.tensor_order > 2:
		raise NotImplementedError()
		
	if not np.all(a.tensor_shape == a.tensor_shape[0]):
		raise ValueError('Need square matrix for determinant.')
	
	#First we need to swap the axes in order to use np.linalg.det
	Temp = np.swapaxes(a, 0, 2)

	return Field(np.linalg.det(Temp), a.grid)

def field_adjoint(a):
	'''Calculates the adjoint of a tensor field.

	Parameters
	----------
	a : Field
		The field for which the adjoint needs to be calculated

	Returns
	-------
	Field
		The adjointed field
	'''
	if a.tensor_order != 2:
		raise ValueError('Only tensor fields of order 2 can be inverted.')
	
	#Calculating the determinant.
	determinant = field_determinant(a)    
	
	if np.any(np.isclose(determinant, 0)):
		raise ValueError('Matrix is non-invertible due to zero determinant.')

	return Field(determinant[np.newaxis,np.newaxis,:] * field_inv(a), a.grid)

def field_cross(a, b):
	'''Calculates the cross product of two vector fields.

	Parameters
	----------
	a : Field
		The first field of the cross product
	b : Field
		The second field of the cross product

	Returns
	-------
	Field
		The cross product field
	'''
	if a.tensor_order != 1 or b.tensor_order != 1:
		raise ValueError('Only tensor fields of order 1 can have a cross product.')

	if a.shape[0] != 3 or b.shape[0] != 3:
		raise ValueError('Vector needs to be of length 3 for cross product.')

	return Field(np.cross(a, b, axis = 0), a.grid)

def make_field_operation(op):
	pass
'''
def make_field_operation(op):
	def inner(*args, **kwargs):
		# Determine which args are fields.
		is_field = [hasattr(arg, 'grid') for arg in args]

		if not np.any(is_field):
			return op(*args, **kwargs)

		# At least one argument has a field
		grid_size = np.flatnonzero(is_field)[0]].grid.size
		
		if len(args) == 1:
			# Only one argument; use loop comprehension
			res = np.array([op(args[0][...,i]) for i in range(grid_size)])

			return Field(, args[0].grid)
		
		# More than one argument operation.
		res = []
		for i in range(grid_size):
			a = tuple([args[j][...,i] if is_field[j] else args[j] for j in len(args)])
			res.append(op(*a, **kwargs))
		return Field(res, )
		'''