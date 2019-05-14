import numpy as np
import copy

from ..field import Field

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
	def __init__(self, psd, output_grid, oversample=1, psd_args=(), psd_kwargs={}):
		from ..fourier import FastFourierTransform

		SpectralNoiseFactory.__init__(self, psd, output_grid, psd_args, psd_kwargs)

		if not self.output_grid.is_regular:
			raise ValueError("Can't make a SpectralNoiseFactoryFFT on a non-regular grid.")
		
		self.fourier = FastFourierTransform(self.output_grid, q=oversample)
		self.input_grid = self.fourier.output_grid
	
		self.period = output_grid.coords.delta * output_grid.coords.shape

		# * (2*np.pi)**self.input_grid.ndim is due to conversion from PSD from "per Hertz" to "per radian", which yields a factor of 2pi per dimension
		self.C = np.sqrt(self.psd(self.input_grid, *psd_args, **psd_kwargs) / self.input_grid.weights * (2*np.pi)**self.input_grid.ndim)

	def make_random(self):
		N = self.input_grid.size
		C = self.C * (np.random.randn(N) + 1j * np.random.randn(N))
		C = Field(C, self.input_grid)

		return SpectralNoiseFFT(self, C)
	
	def make_zero(self):
		C = Field(np.zeros(self.input_grid.size, dtype='complex'), self.input_grid)
		return SpectralNoiseFFT(self, C)
	
class SpectralNoiseFFT(SpectralNoise):
	def __init__(self, factory, C):
		self.factory = factory
		self.C = C
		self.coords = C.grid.separated_coords

	def shift(self, shift):
		s = shift#np.mod(shift + self.factory.period / 2, self.factory.period) - self.factory.period / 2
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
		return self.factory.fourier.backward(self.C).real
	
class SpectralNoiseFactoryMultiscale(SpectralNoiseFactory):
	def __init__(self, psd, output_grid, oversampling, psd_args=(), psd_kwargs={}):
		from ..fourier import FastFourierTransform, MatrixFourierTransform

		SpectralNoiseFactory.__init__(self, psd, output_grid, psd_args, psd_kwargs)

		self.oversampling = oversampling

		self.fourier_1 = FastFourierTransform(self.output_grid)
		self.input_grid_1 = self.fourier_1.output_grid

		self.input_grid_2 = self.input_grid_1.scaled(1.0 / oversampling)
		self.fourier_2 = MatrixFourierTransform(self.output_grid, self.input_grid_2)

		boundary = np.abs(self.input_grid_2.x).max()
		mask_1 = self.input_grid_1.as_('polar').r < boundary
		mask_2 = self.input_grid_2.as_('polar').r >= boundary

		# * (2*np.pi)**self.input_grid.ndim is due to conversion from PSD from "per Hertz" to "per radian", which yields a factor of 2pi per dimension
		self.C_1 = np.sqrt(psd(self.input_grid_1, *psd_args, **psd_kwargs) / self.input_grid_1.weights * (2*np.pi)**self.input_grid_1.ndim)
		self.C_1[mask_1] = 0
		self.C_2 = np.sqrt(psd(self.input_grid_2, *psd_args, **psd_kwargs) / self.input_grid_2.weights * (2*np.pi)**self.input_grid_1.ndim)
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

		C_1 = Field(np.zeros(N_1, dtype='complex'), self.C_1.grid)
		C_2 = Field(np.zeros(N_1, dtype='complex'), self.C_2.grid)

		return SpectralNoiseMultiscale(self, C_1, C_2)
	
class SpectralNoiseMultiscale(SpectralNoise):
	def __init__(self, factory, C_1, C_2):
		self.factory = factory

		self.C_1 = C_1
		self.C_2 = C_2

		self.coords_1 = C_1.grid.separated_coords
		self.coords_2 = C_2.grid.separated_coords

	def shift(self, shift):
		S_1 = [shift[i] * self.coords_1[i] for i in range(len(self.coords_1))]
		S_1 = np.add.reduce(np.ix_(*S_1))

		S_2 = [shift[i] * self.coords_2[i] for i in range(len(self.coords_2))]
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
		ps = self.factory.fourier_1.backward(self.C_1).real
		ps += self.factory.fourier_2.backward(self.C_2).real
		return ps