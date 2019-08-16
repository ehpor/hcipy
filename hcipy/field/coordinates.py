import numpy as np
import copy

class CoordsBase(object):
	'''Base class for coordinates.
	'''

	def copy(self):
		'''Make a copy.
		'''
		return copy.deepcopy(self)
	
	def __add__(self, b):
		'''Add `b` to the coordinates separately and return the result.
		'''
		res = self.copy()
		res += b
		return res
	
	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		raise NotImplementedError()
	
	def __radd__(self, b):
		'''Add `b` to the coordinates separately and return the result.
		'''
		return self + b
	
	def __sub__(self, b):
		'''Subtract `b` from the coordinates separately and return the result.
		'''
		return self + (-b)
	
	def __isub__(self, b):
		'''Subtract `b` from the coordinates separately in-place.
		'''
		self += (-b)
		return self
	
	def __mul__(self, f):
		'''Multiply each coordinate with `f` separately and return the result.
		'''
		res = self.copy()
		res *= f
		return res
	
	def __rmul__(self, f):
		'''Multiply each coordinate with `f` separately and return the result.
		'''
		return self * f
	
	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		raise NotImplementedError()
	
	def __div__(self, f):
		'''Divide each coordinate with `f` separately and return the result.
		'''
		return self * (1./f)
	
	def __idiv__(self, f):
		'''Divide each coordinate with `f` separately in-place.
		'''
		self *= (1./f)
		return self
	
	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		raise NotImplementedError()
	
	@property
	def is_separated(self):
		'''True if the coordinates are separated, False otherwise.
		'''
		return hasattr(self, 'separated_coords')
	
	@property
	def is_regular(self):
		'''True if the coordinates are regularly-spaced, False otherwise.
		'''
		return hasattr(self, 'regular_coords')
	
	@property
	def is_unstructured(self):
		'''True if the coordinates are not structured, False otherwise.
		'''
		return not self.is_separated

	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		raise NotImplementedError()
	
	@property
	def size(self):
		'''The number of points.
		'''
		raise NotImplementedError()
	
	def __len__(self):
		'''The number of dimensions.
		'''
		raise NotImplementedError()

class UnstructuredCoords(CoordsBase):
	'''An unstructured list of points.

	Parameters
	----------
	coords : list or tuple
		A tuple of a list of positions for each dimension.
	'''
	def __init__(self, coords):
		self.coords = list(coords)

	@property
	def size(self):
		'''The number of points.
		'''
		return self.coords[0].size
	
	def __len__(self):
		'''The number of dimensions.
		'''
		return len(self.coords)
	
	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		return self.coords[i]
	
	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		b = np.ones(len(self.coords)) * b
		for i in range(len(self.coords)):
			self.coords[i] += b[i]
		return self
	
	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		f = np.ones(len(self.coords)) * f
		for i in range(len(self.coords)):
			self.coords[i] *= f[i]
		return self
	
	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		for i in range(len(self.coords)):
			self.coords[i] = self.coords[i][::-1]
		return self

class SeparatedCoords(CoordsBase):
	'''A list of points that are separable along each dimension.

	The actual points are given by the iterated tensor product of the `separated_coords`.

	Parameters
	----------
	separated_coords : list or tuple
		A tuple of a list of coordinates along each dimension.

	Attributes
	----------
	separated_coords
		A tuple of a list of coordinates along each dimension.
	'''
	def __init__(self, separated_coords):
		# Make a copy to avoid modification from outside the class
		self.separated_coords = [copy.deepcopy(s) for s in separated_coords]

	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		s0 = (1,) * len(self)
		j = len(self) - i - 1
		output = self.separated_coords[i].reshape(s0[:j] + (-1,) + s0[j + 1:])
		return np.broadcast_to(output, self.shape).ravel()

	@property
	def size(self):
		'''The number of points.
		'''
		return np.prod(self.shape)
	
	def __len__(self):
		'''The number of dimensions.
		'''
		return len(self.separated_coords)
	
	@property
	def dims(self):
		'''The number of points along each dimension.
		'''
		return np.array([len(c) for c in self.separated_coords])
	
	@property
	def shape(self):
		'''The shape of an ``numpy.ndarray`` with the right dimensions.
		'''
		return self.dims[::-1]
	
	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		for i in range(len(self)):
			self.separated_coords[i] += b[i]
		return self
	
	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		if np.isscalar(f):
			for i in range(len(self)):
				self.separated_coords[i] *= f
		else:
			for i in range(len(self)):
				self.separated_coords[i] *= f[i]
		return self
	
	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		for i in range(len(self)):
			self.separated_coords[i] = self.separated_coords[i][::-1]
		return self

class RegularCoords(CoordsBase):
	'''A list of points that have a regular spacing in all dimensions.

	Parameters
	----------
	delta : array_like
		The spacing between the points.
	dims : array_like
		The number of points along each dimension.
	zero : array_like
		The coordinates for the first point.

	Attributes
	----------
	delta
		The spacing between the points.
	dims
		The number of points along each dimension.
	zero
		The coordinates for the first point.
	'''
	def __init__(self, delta, dims, zero=None):
		if np.isscalar(dims):
			self.dims = np.array([dims]).astype('int')
		else:
			self.dims = np.array(dims).astype('int')
		
		if np.isscalar(delta):
			self.delta = np.array([delta]*len(self.dims))
		else:
			self.delta = np.array(delta)

		if zero is None:
			self.zero = np.zeros(len(self.dims))
		elif np.isscalar(zero):
			self.zero = np.array([zero]*len(self.dims))
		else:
			self.zero = np.array(zero)

	@property
	def separated_coords(self):
		'''A tuple of a list of the values for each dimension.

		The actual points are the iterated tensor product of this tuple.
		'''
		return [np.arange(n) * delta + zero for delta, n, zero in zip(self.delta, self.dims, self.zero)]

	@property
	def regular_coords(self):
		'''The tuple `(delta, dims, zero)` of the regularly-spaced coordinates.
		'''
		return self.delta, self.dims, self.zero

	@property
	def size(self):
		'''The number of points.
		'''
		return np.prod(self.dims)
	
	def __len__(self):
		'''The number of dimensions.
		'''
		return len(self.dims)
	
	@property
	def shape(self):
		'''The shape of an ``numpy.ndarray`` with the right dimensions.
		'''
		return self.dims[::-1]

	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		s0 = (1,) * len(self)
		j = len(self) - i - 1
		t = s0[:j] + (-1,) + s0[j + 1:]
		output = self.separated_coords[i].reshape(t)
		return np.broadcast_to(output, self.shape).ravel()
	
	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		self.zero += b
		return self
	
	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		self.delta *= f
		self.zero *= f
		return self
	
	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		maximum = self.zero + self.delta * (self.dims - 1)
		self.delta = -self.delta
		self.zero = maximum
		return self