import numpy as np
import string

class Field(np.ndarray):
	'''The value of some physical quantity for each point in some coordinate system.

	Parameters
	----------
	arr : array_like
		An array of values or tensors for each point in the :class:`Grid`.
	grid : Grid
		The corresponding :class:`Grid` on which the values are set.
	
	Attributes
	----------
	grid : Grid
		The grid on which the values are defined.

	'''
	def __new__(cls, arr, grid):
		obj = np.asarray(arr).view(cls)
		obj.grid = grid
		return obj
	
	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.grid = getattr(obj, 'grid', None)
	
	@property
	def tensor_order(self):
		'''The order of the tensor of the field.
		'''
		return self.ndim - 1

	@property
	def tensor_shape(self):
		'''The shape of the tensor of the field.
		'''
		return np.array(self.shape)[:-1]
	
	@property
	def is_scalar_field(self):
		'''True if this field is a scalar field (ie. a tensor order of 0), False otherwise.
		'''
		return self.tensor_order == 0
	
	@property
	def is_vector_field(self):
		'''True if this field is a vector field (ie. a tensor order of 1), False otherwise.
		'''
		return self.tensor_order == 1
	
	@property
	def is_valid_field(self):
		'''True if the field corresponds with its grid.
		'''
		return self.shape[-1] == self.grid.size
	
	@property
	def shaped(self):
		'''The reshaped version of this field.

		Raises
		------
		ValueError
			If this field isn't separated, no reshaped version can be made.
		'''
		if not self.grid.is_separated:
			raise ValueError('This field doesn\'t have a shape.')

		if self.tensor_order > 0:
			new_shape = np.concatenate([np.array(self.shape)[:-1], self.grid.shape])
			return self.reshape(new_shape)

		return self.reshape(self.grid.shape)
	
	def at(self, p):
		'''The value of this field closest to point p.

		Parameters
		----------
		p : array_like
			The point at which the closest value should be returned.
		
		Returns
		-------
		array_like
			The value, potentially tensor, closest to point p.

		'''
		i = self.grid.closest_to(p)
		return self[...,i]

def field_einsum(subscripts, *operands, **kwargs):
	'''Evaluates the Einstein summation convention on the operand fields.

	This function uses the same conventions as numpy.einsum(). The input
	subscript is multiplexed over each position in the grid. The grids of each of
	the input field operands don't have to match, but must have the same lengths.

	The subscripts must be written as you would for a single position in the grid.
	The function alters these subscripts to multiplex over the entire grid.

	.. caution::
		Some subscripts may yield no exception, even though they would fail for
		a single point in the grid. The output in these cases can not be trusted.

	Parameters
	----------
	subscripts : str
		Specifies the subscripts for summation.
	operands : list of array_like or `Field`
		These are the arrays or fields for the operation.
	out : {ndarray, None}, optional
		If provided, the calculation is done into this array.
	dtype : {data-type, None}, optional
		If provided, forces the calculation to use the data type specified.
		Note that you may have to also give a more liberal `casting`
		parameter to allow the conversions. Default is None.
	order : {'C', 'F', 'A', 'K'}, optional
		Controls the memory layout of the output. 'C' means it should
		be C contiguous. 'F' means it should be Fortran contiguous,
		'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
		'K' means it should be as close to the layout as the inputs as
		is possible, including arbitrarily permuted axes.
		Default is 'K'.
	casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
		Controls what kind of data casting may occur.  Setting this to
		'unsafe' is not recommended, as it can adversely affect accumulations.

			* 'no' means the data types should not be cast at all.
			* 'equiv' means only byte-order changes are allowed.
			* 'safe' means only casts which can preserve values are allowed.
			* 'same_kind' means only safe casts or casts within a kind,	
				like float64 to float32, are allowed.
			* 'unsafe' means any data conversions may be done.

		Default is 'safe'.
	optimize : {False, True, 'greedy', 'optimal'}, optional
		Controls if intermediate optimization should occur. No optimization
		will occur if False and True will default to the 'greedy' algorithm.
		Also accepts an explicit contraction list from the ``np.einsum_path``
		function. See ``np.einsum_path`` for more details. Default is False.
	
	Returns
	-------
	Field
		The calculated Field based on the Einstein summation convention.
	
	Raises
	------
	ValueError
		If all of the fields don't have	the same grid size. If the number of
		operands is not equal to the number of subscripts specified.
	'''
	is_field = [isinstance(o, Field) for o in operands]
	if not np.count_nonzero(is_field):
		return np.einsum(subscripts, *operands, **kwargs)
	
	field_sizes = [o.grid.size for i, o in enumerate(operands) if is_field[i]]
	if not np.allclose(field_sizes, field_sizes[0]):
		raise ValueError('All fields must be the same size for a field_einsum().')

	# Decompose the subscript into input and output
	splitted_string = subscripts.split('->')
	if len(splitted_string) == 2:
		ss_input, ss_output = splitted_string
	else:
		ss_input = splitted_string[0]
		ss_output = ''
	
	# split the input operands in separate strings
	ss = ss_input.split(',')
	if len(ss) != len(operands):
		raise ValueError('Number of operands is not equal to number of indexing operands.')
	
	# Find an indexing letter that can be used for field dimension.
	unused_index = [a for a in string.ascii_lowercase if a not in subscripts][0]

	# Add the field dimension to the input field operands.
	ss = [s + unused_index if is_field[i] else s for i,s in enumerate(ss)]

	# Recombine all operands into the final subscripts
	if len(splitted_string) == 2:
		subscripts_new = ','.join(ss) + '->' + ss_output + unused_index
	else:
		subscripts_new = ','.join(ss)

	res = np.einsum(subscripts_new, *operands, **kwargs)
	grid = operands[np.flatnonzero(np.array(is_field))[0]].grid

	if 'out' in kwargs:
		kwargs['out'] = Field(res, grid)
	return Field(res, grid)

def field_dot(a, b, out=None):
	'''Perform a dot product of `a` and `b` multiplexed over the field dimension.

	Parameters
	----------
	a : Field or array_like
		Left argument of the dot product.
	b : Field or array_like
		Right argument of the dot product.
	out : Field or array_like
		If provided, the calculation is done into this array.

	Returns
	-------
	Field
		The result of the dot product.
	'''
	# Find out if a or b are vectors or higher dimensional tensors
	if hasattr(a, 'tensor_order'):
		amat = a.tensor_order > 1
	elif np.isscalar(a):
		if out is None:
			return a * b
		else:
			return np.multiply(a, b, out)
	else:
		amat = a.ndim > 1
	
	if hasattr(b, 'tensor_order'):
		bmat = b.tensor_order > 1
	elif np.isscalar(b):
		if out is None:
			return a * b
		else:
			return np.multiply(a, b, out)
	else:
		bmat = b.ndim > 1
	
	# Select correct multiplication behaviour.
	if amat and bmat:
		subscripts = '...ij,...jk->...ik'
	elif amat and not bmat:
		subscripts = '...i,...i->...'
	elif not amat and bmat:
		subscripts = '...i,...ij->...j'
	elif not amat and not bmat:
		subscripts = '...i,...i->...'

	# Perform calculation and return.
	if out is None:
		return field_einsum(subscripts, a, b)
	else:
		return field_einsum(subscripts, a, b, out=out)

def field_trace(a, out=None):
	if out is None:
		return field_einsum('ii', a)
	else:
		return field_einsum('ii', a, out=out)

def field_inv(a):
	if hasattr(a, 'grid'):
		if a.tensor_order != 2:
			raise ValueError("Only tensor fields of order 2 can be inverted.")
		
		res = np.rollaxis(np.linalg.inv(np.rollaxis(a, -1)), 0, 3)
		return Field(res, a.grid)
	else:
		return np.linalg.inv(a)