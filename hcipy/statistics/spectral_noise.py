import numpy as np
import copy

class SpectralNoiseFactory(object):
	def __init__(self, psd, output_grid, psd_args=(), psd_kwargs={}):
		self.psd = psd
		self.psd_args = psd_args
		self.psd_kwargs = psd_kwargs
		self.output_grid = output_grid
	
	def make_random(self):
		raise NotImplementedError()
	
	def make_zero(self):
		raise NotImplementedError()

class SpectralNoise(object):
	def copy(self):
		return copy.deepcopy(self)
	
	def shift(self, shift):
		raise NotImplementedError()
	
	def shifted(self, shift):
		a = self.copy()
		a.shift(shift)
		return a
	
	def __iadd__(self, b):
		return NotImplemented
	
	def __add__(self, b):
		a = self.copy()
		a += b
		return a
	
	def __imul__(self, f):
		return NotImplemented
	
	def __mul__(self, f):
		a = self.copy()
		a *= f
		return a

	def __call__(self):
		raise NotImplementedError()

	def evaluate(self):
		return self()

class SpectralNoiseFactoryFFT(SpectralNoiseFactory):
	def __init__(self, psd, output_grid, psd_args=(), psd_kwargs={}):
		from ..fourier import FastFourierTransform

		SpectralNoiseFactory.__init__(self, psd, output_grid, psd_args, psd_kwargs)

		if not self.output_grid.is_regular:
			raise ValueError("Can't make a SpectralNoiseFactoryFFT on a non-regular grid.")
		
		self.fourier = FastFourierTransform(self.output_grid)
		self.input_grid = self.fourier.output_grid
	
		self.period = output_grid.coords.delta * output_grid.coords.shape

		self.C = np.sqrt(self.psd(self.input_grid, *psd_args, **psd_kwargs) / self.input_grid.weights)

	def make_random(self):
		N = self.input_grid.size
		C = self.C * (np.random.randn(N) + 1j * np.random.randn(N))

		return SpectralNoiseFFT(self, C)
	
	def make_zero(self):
		return SpectralNoise(self, np.zeros(self.input_grid.size, dtype='complex'))
	
class SpectralNoiseFFT(SpectralNoise):
	def __init__(self, factory, C):
		self.factory = factory
		self.C = C
		self.coords = C.grid.separated_coords

	def shift(self, shift):
		s = np.mod(shift + self.factory.period / 2, self.factory.period) - self.factory.period / 2
		S = [s[i] * self.coords[i] for i in range(len(self.coords))]
		S = np.add.reduce(np.ix_(*S))
		self.C *= np.exp(-1j * S.ravel())
	
	def __iadd__(self, b):
		self.C += b.C
		return self
	
	def __imul__(self, f):
		self.C *= f
		return self
	
	def __call__(self):
		return self.factory.fourier.inverse(self.C).real
	
class SpectralNoiseFactoryMultiscale(SpectralNoiseFactory):
	def __init__(self, psd, output_grid, oversampling, psd_args=(), psd_kwargs={}):
		SpectralNoiseFactory.__init__(self, psd, output_grid, psd_args, psd_kwargs)

		self.oversampling = oversampling

		self.fourier_1 = FastFourierTransform(self.output_grid)
		self.input_grid_1 = self.fourier_1.output_grid

		self.fourier_2 = self.input_grid_1.scale(1 / oversampling)
		self.fourier_2 = MatrixFourierTransform(self.output_grid, self.input_grid_2)

		boundary = np.abs(self.input_grid_2.x).max()
		mask_1 = self.input_grid_1.as_('polar').r < boundary
		mask_2 = self.input_grid_2.as_('polar').r >= boundary

		self.C_1 = np.sqrt(self.input_grid_1, *psd_args, **psd_kwargs) / self.input_grid_1.weights
		self.C_1[mask_1] = 0
		self.C_2 = np.sqrt(self.input_grid_2, *psd_args, **psd_kwargs) / self.input_grid_2.weights
		self.C_2[mask_2] = 0
	
	def make_random(self):
		N_1 = self.input_grid_1.size
		N_2 = self.input_grid_2.size
		
		C_1 = self.C_1 * (np.random.randn(N_1) + 1j * np.random.randn(N_1))
		C_2 = self.C_2 * (np.random.randn(N_2) + 1j * np.random.randn(N_2))

		return SpectralNoiseMultiscale(self, C_1, C_2)
	
	def make_zero(self):
		N_1 = self.input_grid_1.size
		N_2 = self.input_grid_2.size

		return SpectralNoiseMultiscale(self, np.zeros(N_1, dtype='complex'), np.zeros(N_2, dtype='complex'))
	
class SpectralNoiseMultiscale(SpectralNoise):
	def __init__(self, factory, C_1, C_2):
		self.factory = factory

		self.C_1 = C_1
		self.C_2 = C_2

		self.coords_1 = C_1.grid.separated_coords
		self.coords_2 = C_2.grid.separated_coords

	def shift(self, shift):
		S_1 = [s[i] * self.coords_1[i] for i in range(len(self.coords_1))]
		S_1 = np.add.reduce(np.ix_(*S_1))

		S_2 = [s[i] * self.coords_2[i] for i in range(len(self.coords_2))]
		S_2 = np.add.reduce(np.ix_(*S_2))

		self.C_1 *= np.exp(-1j * S_1.ravel())
		self.C_2 *= np.exp(-1j * S_2.ravel())
	
	def __iadd__(self, b):
		self.C_1 += b.C_1
		self.C_2 += b.C_2
		return self
	
	def __imul__(self, f):
		self.C_1 *= f
		self.C_2 *= f
		return self

	def __call__(self):
		ps = self.factory.fourier_1.inverse(self.C_1).real
		ps += self.factory.fourier_2.inverse(self.C_2).real
		return ps