import numpy as np
import copy

class CoordsBase(object):
	def copy(self):
		return copy.deepcopy(self)
	
	def __add__(self, b):
		res = self.copy()
		res += b
		return res
	
	def __radd__(self, b):
		return self + b
	
	def __sub__(self, b):
		return self + (-b)
	
	def __isub__(self, b):
		self += (-b)
		return self
	
	def __mul__(self, f):
		res = self.copy()
		res *= f
		return res
	
	def __rmul__(self, f):
		return self * f
	
	def __div__(self, f):
		return self * (1./f)
	
	def __idiv__(self, f):
		self *= (1./f)
		return self
	
	@property
	def is_separated(self):
		return hasattr(self, 'separated_coords')
	
	@property
	def is_regular(self):
		return hasattr(self, 'regular_coords')

class UnstructuredCoords(CoordsBase):
	def __init__(self, coords):
		self.coords = list(coords)

	@property
	def size(self):
		return self.coords[0].size
	
	def __len__(self):
		return len(self.coords)
	
	def __getitem__(self, i):
		return self.coords[i]
	
	def __iadd__(self, b):
		b = np.ones(len(self.coords)) * b
		for i in range(len(self.coords)):
			self.coords[i] += b[i]
		return self
	
	def __imul__(self, f):
		f = np.ones(len(self.coords)) * f
		for i in range(len(self.coords)):
			self.coords[i] *= f[i]
		return self
	
	def reverse(self):
		for i in range(len(self.coords)):
			self.coords[i] = self.coords[i][::-1]
		return self

class SeparatedCoords(CoordsBase):
	def __init__(self, separated_coords):
		self.separated_coords = list(separated_coords)

	def __getitem__(self, i):
		return np.meshgrid(*self.separated_coords)[i].ravel()

	@property
	def size(self):
		return np.prod(self.shape)
	
	def __len__(self):
		return len(self.separated_coords)
	
	@property
	def shape(self):
		return np.array([len(c) for c in self.separated_coords])
	
	def __iadd__(self, b):
		for i in range(len(self)):
			self.separated_coords[i] += b[i]
		return self
	
	def __imul__(self, f):
		if np.isscalar(f):
			for i in range(len(self)):
				self.separated_coords[i] *= f
		else:
			for i in range(len(self)):
				self.separated_coords[i] *= f[i]
		return self
	
	def reverse(self):
		for i in range(len(self)):
			self.separated_coords[i] = self.separated_coords[i][::-1]
		return self

class RegularCoords(CoordsBase):
	def __init__(self, delta, shape, zero=None):
		if np.isscalar(delta):
			self.delta = np.array([delta])
		else:
			self.delta = np.array(delta)

		if np.isscalar(shape):
			self.shape = np.array([shape]).astype('int')
		else:
			self.shape = np.array(shape).astype('int')

		if zero is None:
			self.zero = np.zeros(len(self.delta))
		elif np.isscalar(zero):
			self.zero = np.array([zero])
		else:
			self.zero = np.array(zero)

	@property
	def separated_coords(self):
		return [np.arange(n) * delta + zero for delta, n, zero in zip(self.delta, self.shape, self.zero)]

	@property
	def regular_coords(self):
		return self.delta, self.shape, self.zero

	@property
	def size(self):
		return np.prod(self.shape)
	
	def __len__(self):
		return len(self.shape)

	def __getitem__(self, i):
		return np.meshgrid(*self.separated_coords)[i].ravel()
	
	def __iadd__(self, b):
		self.zero += b
		return self
	
	def __imul__(self, f):
		self.delta *= f
		self.zero *= f
		return self
	
	def reverse(self):
		maximum = self.zero + self.delta * (self.shape - 1)
		self.delta = -self.delta
		self.zero = maximum
		return self
