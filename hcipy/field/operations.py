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