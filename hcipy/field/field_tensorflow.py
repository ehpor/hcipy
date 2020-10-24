from hcipy.field.field_base import field_backend, FieldBase

import numpy as np
import numpy.lib.mixins

_tf_handled_functions = {}
_tf_handled_ufuncs = {}

def _implements_ufunc(numpy_ufunc):
	def decorator(func):
		_tf_handled_ufuncs[numpy_ufunc] = func
		return func
	return decorator

def _implements_function(numpy_function):
	def decorator(func):
		_tf_handled_functions[numpy_function] = func
		return func
	return decorator

def _unwrap_and_dtype_cast(*inputs):
	import tensorflow as tf

	grid = None
	new_inputs = []
	dtypes = []

	for a in inputs:
		if hasattr(a, 'arr'):
			grid = a.grid
			new_inputs.append(a.arr)

			if tf.is_tensor(a.arr):
				dtypes.append(a.arr.dtype.as_numpy_dtype)
			else:
				dtypes.append(a)
		elif tf.is_tensor(a):
			new_inputs.append(a)
			dtypes.append(a.dtype.as_numpy_dtype)
		else:
			new_inputs.append(a)
			dtypes.append(a)

	result_type = np.result_type(*dtypes)
	new_inputs = tuple(tf.cast(a, result_type) for a in new_inputs)

	return new_inputs, grid

def _unary_op(tf_ufunc):
	def op(a):
		return TensorFlowField(tf_ufunc(a.arr), a.grid)
	return op

def _binary_op(tf_ufunc):
	def op(a, b, out=None):
		(a, b), grid = _unwrap_and_dtype_cast(a, b)
		res = tf_ufunc(a, b)

		if out is not None:
			out[0].arr = res
			out[0].grid = grid

		return TensorFlowField(res, grid)
	return op

def _reduce_function(tf_function):
	def func(a, **kwargs):
		res = tf_function(a.arr, **kwargs)

		if res.ndim == 0:
			return res
		return TensorFlowField(res, a.grid)
	return func

def _sequence_function(tf_function):
	def func(sequence, axis=0):
		sequence, grid = _unwrap_and_dtype_cast(*sequence)

		return TensorFlowField(tf_function(sequence, axis=axis), grid)
	return func

def _initialize_functions_and_ufuncs():
	import tensorflow as tf

	ufuncs_unary = [
		('sin', tf.sin),
		('cos', tf.cos),
		('tan', tf.tan),
		('arcsin', tf.asin),
		('arccos', tf.acos),
		('arctan', tf.atan),
		('arctan2', tf.atan2),
		('degrees', lambda a: a * (180 / np.pi)),
		('radians', lambda a: a * (np.pi / 180)),
		('unwrap', None),
		('deg2rad', lambda a: a * (np.pi / 180)),
		('rad2deg', lambda a: a * (180 / np.pi)),
		('sinh', tf.sinh),
		('cosh', tf.cosh),
		('tanh', tf.tanh),
		('arcsinh', tf.asinh),
		('arccosh', tf.acosh),
		('arctanh', tf.atanh),
		('exp', tf.exp),
		('expm1', tf.math.expm1),
		('exp2', lambda a: tf.pow(2, a)),
		('log', tf.math.log),
		('log10', lambda a: tf.math.log(a) / np.log(10)),
		('log2', lambda a: tf.math.log(a) / np.log(2)),
		('log1p', tf.math.log1p),
		('angle', tf.math.angle),
		('conj', tf.math.conj),
		('conjugate', tf.math.conj),
		('cbrt', lambda a: tf.pow(a, 1 / 3)),
		('square', tf.square),
		('absolute', tf.abs),
		('abs', tf.abs),
		('sign', tf.sign),
		('reciprocal', tf.math.reciprocal),
		('positive', lambda a: a),
		('negative', tf.negative),
		('floor', tf.floor),
		('ceil', tf.math.ceil)
	]

	ufuncs_binary = [
		('add', tf.add),
		('subtract', tf.subtract),
		('multiply', tf.multiply),
		('divide', tf.divide),
		('true_divide', tf.truediv),
		('floor_divide', tf.math.floordiv),
		('power', tf.pow),
		('hypot', lambda a, b: tf.sqrt(a**2 + b**2)),
		('less', tf.less),
		('less_equal', tf.less_equal),
		('greater', tf.greater),
		('greater_equal', tf.greater_equal),
		('equal', tf.equal),
		('not_equal', tf.not_equal)
	]

	functions_reduce = [
		('var', tf.math.reduce_variance),
		('sum', tf.math.reduce_sum),
		('std', tf.math.reduce_std),
		('prod', tf.math.reduce_prod),
		('min', tf.math.reduce_min),
		('max', tf.math.reduce_max),
		('mean', tf.math.reduce_mean),
		('amax', tf.math.reduce_max),
		('amin', tf.math.reduce_min),
		('argmax', tf.math.argmax),
		('argmin', tf.math.argmin),
		('cumsum', tf.math.cumsum),
		('cumprod', tf.math.cumprod)
	]

	functions_sequence = [
		('concatenate', tf.concat),
		('stack', tf.stack)
	]

	functions_complex = [
		('real', tf.math.real),
		('imag', tf.math.imag)
	]

	for ufunc, tf_func in ufuncs_unary:
		_implements_ufunc(ufunc)(_unary_op(tf_func))

	for ufunc, tf_func in ufuncs_binary:
		_implements_ufunc(ufunc)(_binary_op(tf_func))

	for func, tf_func in functions_reduce:
		_implements_function(func)(_reduce_function(tf_func))

	for func, tf_func in functions_sequence:
		_implements_function(func)(_sequence_function(tf_func))

	for func, tf_func in functions_complex:
		_implements_function(func)(_unary_op(tf_func))

@_implements_function('dot')
def _tf_dot(a, b):
	import tensorflow as tf

	(a, b), grid = _unwrap_and_dtype_cast(a, b)

	if np.isscalar(a) or np.isscalar(b):
		return TensorFlowField(tf.multiply(a, b), grid)

	if a.ndim == 1 and b.ndim == 1:
		return tf.math.reduce_sum(tf.multiply(a, b))

	if a.ndim > 1 and b.ndim == 1:
		return TensorFlowField(tf.matvec(a, b), grid)

	return TensorFlowField(tf.matmul(a, b), grid)

@_implements_function('transpose')
def _tf_transpose(a, axes=None):
	import tensorflow as tf

	return TensorFlowField(tf.tranpose(a.arr, perm=axes), a.grid)

@_implements_function('reshape')
def _tf_reshape(a, newshape):
	import tensorflow as tf

	return TensorFlowField(tf.reshape(a.arr, newshape), a.grid)

@_implements_function('ravel')
def _tf_ravel(a):
	import tensorflow as tf

	return TensorFlowField(tf.reshape(a.arr, (-1,)), a.grid)

@_implements_function('einsum')
def _tf_einsum(subscripts, *inputs, **kwargs):
	import tensorflow as tf

	new_inputs, grid = _unwrap_and_dtype_cast(*inputs)

	res = tf.einsum(subscripts, *new_inputs, **kwargs)

	if res.ndim == 0:
		return res
	return TensorFlowField(res, grid)

@_implements_function('clip')
def _tf_clip(a, a_min, a_max):
	import tensorflow as tf

	(a, a_min, a_max), grid = _unwrap_and_dtype_cast(a, a_min, a_max)

	return TensorFlowField(tf.clip_by_value(a, a_min, a_max), grid)

@field_backend('tensorflow')
class TensorFlowField(FieldBase, numpy.lib.mixins.NDArrayOperatorsMixin):
	_initialized_class = False

	def __init__(self, arr, grid):
		import tensorflow as tf

		if not self.__class__._initialized_class:
			_initialize_functions_and_ufuncs()
			self.__class__._initialized_class = True

		if hasattr(arr, 'arr'):
			arr = arr.arr

		if not tf.is_tensor(arr):
			arr = tf.convert_to_tensor(arr)

		self.arr = arr
		self.grid = grid

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		print('__array_ufunc__', ufunc, method)

		if method != '__call__':
			return NotImplemented

		name = ufunc.__name__

		if name not in _tf_handled_ufuncs:
			return NotImplemented

		return _tf_handled_ufuncs[name](*inputs, **kwargs)

	def __array_function__(self, func, types, args, kwargs):
		print('__array_function__', func)

		name = '.'.join(func.__module__.split('.')[1:] + [func.__name__])

		if name not in _tf_handled_functions:
			return NotImplemented

		return _tf_handled_functions[name](*args, **kwargs)

	@property
	def T(self):
		return np.tranpose(self)

	@property
	def dtype(self):
		return np.dtype(self.arr.dtype.as_numpy_dtype)

	@property
	def imag(self):
		return np.imag(self)

	@property
	def real(self):
		return np.real(self)

	@property
	def size(self):
		return np.prod(self.shape)

	@property
	def ndim(self):
		return len(self.shape)

	@property
	def shape(self):
		return self.arr.shape

	def astype(self, dtype, *args, **kwargs):
		import tensorflow as tf

		return self.__class__(tf.cast(self.arr, tf.as_dtype(np.dtype(dtype))), self.grid)

	@classmethod
	def _from_numpy(cls, field):
		import tensorflow as tf

		return cls(tf.convert_to_tensor(field), field.grid)

	def _to_numpy(self):
		import tensorflow as tf
		from .field_numpy import NumpyField

		return NumpyField(self.arr.numpy(), self.grid)

	all = np.all
	any = np.any
	argmax = np.argmax
	argmin = np.argmin
	argpartition = np.argpartition
	argsort = np.argsort
	clip = np.clip
	compress = np.compress
	conj = np.conj
	conjugate = np.conjugate
	copy = np.copy
	cumprod = np.cumprod
	cumsum = np.cumsum
	dot = np.dot
	flatten = np.ravel
	max = np.max
	mean = np.mean
	min = np.min
	nonzero = np.nonzero
	prod = np.prod
	ptp = np.ptp
	ravel = np.ravel
	repeat = np.repeat
	reshape = np.reshape
	round = np.round
	sort = np.sort
	squeeze = np.squeeze
	std = np.std
	sum = np.sum
	trace = np.trace
	tranpose = np.transpose
	var = np.var
