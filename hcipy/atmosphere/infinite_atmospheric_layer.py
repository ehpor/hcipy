from __future__ import division

from .atmospheric_model import AtmosphericLayer, phase_covariance_von_karman, fried_parameter_from_Cn_squared
from ..statistics import SpectralNoiseFactoryMultiscale
from ..field import Field, RegularCoords, UnstructuredCoords, CartesianGrid
from .finite_atmospheric_layer import FiniteAtmosphericLayer

import numpy as np
from scipy import linalg
from scipy.ndimage import affine_transform

import time
import warnings

class InfiniteAtmosphericLayer(AtmosphericLayer):
	def __init__(self, input_grid, Cn_squared=None, L0=np.inf, velocity=0, height=0, stencil_length=2, use_interpolation=False):
		self._initialized = False

		AtmosphericLayer.__init__(self, input_grid, Cn_squared, L0, velocity, height)

		# Check properties of input_grid
		if not input_grid.is_('cartesian'):
			raise ValueError('Input grid must be cartesian.')
		if not input_grid.is_regular:
			raise ValueError('Input grid must be regularly spaced')
		if not input_grid.ndim == 2:
			raise ValueError('Input grid must be two-dimensional.')

		self.stencil_length = stencil_length
		self.use_interpolation = use_interpolation

		self._make_stencils()
		self._make_covariance_matrices()
		self._make_AB_matrices()
		self._make_initial_phase_screen()

		self.center = np.zeros(2)

		self._initialized = True
	
	def _recalculate_matrices(self):
		if self._initialized:
			self._make_covariance_matrices()
			self._make_AB_matrices()
	
	def _make_stencils(self):
		# Vertical
		self.new_grid_bottom = CartesianGrid(RegularCoords(self.input_grid.delta, [self.input_grid.dims[0], 1], self.input_grid.zero - np.array([0, self.input_grid.delta[1]])))
		
		self.stencil_bottom = Field(np.zeros(self.input_grid.size, dtype='bool'), self.input_grid).shaped
		self.stencil_bottom[:self.stencil_length,:] = True
		
		for i, n in enumerate(np.random.geometric(0.5, self.input_grid.dims[0])):
			self.stencil_bottom[(n + self.stencil_length - 1) % self.input_grid.dims[1],i] = True
		
		self.stencil_bottom = self.stencil_bottom.ravel()
		self.num_stencils_vertical = np.sum(self.stencil_bottom)
		
		# Horizontal
		self.new_grid_left = CartesianGrid(RegularCoords(self.input_grid.delta, [1, self.input_grid.dims[1]], self.input_grid.zero - np.array([self.input_grid.delta[0], 0])))

		self.stencil_left = Field(np.zeros(self.input_grid.size, dtype='bool'), self.input_grid).shaped
		self.stencil_left[:,:self.stencil_length] = True
		
		for i, n in enumerate(np.random.geometric(0.5, self.input_grid.dims[1])):
			self.stencil_left[i,(n + self.stencil_length - 1) % self.input_grid.dims[0]] = True
		
		self.stencil_left = self.stencil_left.ravel()
		self.num_stencils_horizontal = np.sum(self.stencil_left)
	
	def _make_covariance_matrices(self):
		phase_covariance = phase_covariance_von_karman(fried_parameter_from_Cn_squared(1, 1), self.L0)

		# Vertical
		x = np.concatenate((self.input_grid.x[self.stencil_bottom], self.new_grid_bottom.x))
		x = np.concatenate([x - xx for xx in x])
		y = np.concatenate((self.input_grid.y[self.stencil_bottom], self.new_grid_bottom.y))
		y = np.concatenate([y - yy for yy in y])

		separations = CartesianGrid(UnstructuredCoords((x, y)))
		n = self.new_grid_bottom.size + self.num_stencils_vertical
		self.cov_matrix_vertical = phase_covariance(separations).reshape((n, n))

		# Horizontal
		x = np.concatenate((self.input_grid.x[self.stencil_left], self.new_grid_left.x))
		x = np.concatenate([x - xx for xx in x])
		y = np.concatenate((self.input_grid.y[self.stencil_left], self.new_grid_left.y))
		y = np.concatenate([y - yy for yy in y])

		separations = CartesianGrid(UnstructuredCoords((x, y)))
		n = self.new_grid_left.size + self.num_stencils_horizontal
		self.cov_matrix_horizontal = phase_covariance(separations).reshape((n, n))
	
	def _make_AB_matrices(self):
		# Vertical
		n = self.num_stencils_vertical
		cov_zz = self.cov_matrix_vertical[:n,:n]
		cov_xz = self.cov_matrix_vertical[n:, :n]
		cov_zx = self.cov_matrix_vertical[:n, n:]
		cov_xx = self.cov_matrix_vertical[n:, n:]
		
		cf = linalg.cho_factor(cov_zz)
		inv_cov_zz = linalg.cho_solve(cf, np.eye(cov_zz.shape[0]))

		self.A_vertical = cov_xz.dot(inv_cov_zz)

		BBt = cov_xx - self.A_vertical.dot(cov_zx)

		U, S, Vt = np.linalg.svd(BBt)
		L = np.sqrt(S[:self.input_grid.dims[0]])

		self.B_vertical = U * L
		
		# Horizontal
		n = self.num_stencils_horizontal
		cov_zz = self.cov_matrix_horizontal[:n,:n]
		cov_xz = self.cov_matrix_horizontal[n:, :n]
		cov_zx = self.cov_matrix_horizontal[:n, n:]
		cov_xx = self.cov_matrix_horizontal[n:, n:]
		
		cf = linalg.cho_factor(cov_zz)
		inv_cov_zz = linalg.cho_solve(cf, np.eye(cov_zz.shape[0]))

		self.A_horizontal = cov_xz.dot(inv_cov_zz)

		BBt = cov_xx - self.A_horizontal.dot(cov_zx)

		U, S, Vt = np.linalg.svd(BBt)
		L = np.sqrt(S[:self.input_grid.dims[1]])

		self.B_horizontal = U * L

	def _make_initial_phase_screen(self):
		oversampling = 16
		layer = FiniteAtmosphericLayer(self.input_grid, self.Cn_squared, self.outer_scale, self.velocity, self.height, oversampling)
		self._achromatic_screen = layer.phase_for(1)
		self._shifted_achromatic_screen = self._achromatic_screen

	def _extrude(self, where=None):
		flipped = (where == 'top') or (where == 'right')
		horizontal = (where == 'left') or (where == 'right')

		if where == 'top' or where == 'right':
			screen = self._achromatic_screen[::-1]
		else:
			screen = self._achromatic_screen

		if horizontal:
			stencil = self.stencil_left
			A = self.A_horizontal
			B = self.B_horizontal
		else:
			stencil = self.stencil_bottom
			A = self.A_vertical
			B = self.B_vertical
		
		stencil_data = screen[stencil]
		random_data = np.random.normal(0, 1, size=B.shape[1])
		new_slice = A.dot(stencil_data) + B.dot(random_data) * np.sqrt(self._Cn_squared)

		screen = screen.shaped

		if horizontal:
			screen = np.hstack((new_slice[:,np.newaxis], screen[:,:-1]))
		else:
			screen = np.vstack((new_slice[np.newaxis,:], screen[:-1,:]))
		
		screen = Field(screen, self.input_grid)
		
		if flipped:
			self._achromatic_screen = screen[::-1,::-1].ravel()
		else:
			self._achromatic_screen = screen.ravel()

	def phase_for(self, wavelength):
		return self._shifted_achromatic_screen / wavelength

	def reset(self):
		self._make_initial_phase_screen()
		self.center = np.zeros(2)
		self._t = 0
	
	def evolve_until(self, t):
		if t is None:
			self.reset()
			return
		
		old_center = np.round(self.center / self.input_grid.delta).astype('int')

		self.center = self.velocity * t
		new_center = np.round(self.center / self.input_grid.delta).astype('int')

		delta = new_center - old_center

		for i in range(abs(delta[0])):
			if delta[0] < 0:
				self._extrude('left')
			else:
				self._extrude('right')

		for i in range(abs(delta[1])):
			if delta[1] < 0:
				self._extrude('bottom')
			else:
				self._extrude('top')
		
		if self.use_interpolation:
			# Use bilinear interpolation to interpolate the achromatic phase screen to the correct position.
			# This is to avoid sudden shifts by discrete pixels.
			ps = self._achromatic_screen.shaped
			sub_delta = self.center - new_center * self.input_grid.delta
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', message='The behaviour of affine_transform with a one-dimensional array supplied for the matrix parameter has changed in scipy 0.18.0.')
				self._shifted_achromatic_screen = affine_transform(ps, np.array([1,1]), (sub_delta / self.input_grid.delta)[::-1], mode='nearest', order=5).ravel()
		else:
			self._shifted_achromatic_screen = self._achromatic_screen

	@property
	def Cn_squared(self):
		return self._Cn_squared
	
	@Cn_squared.setter
	def Cn_squared(self, Cn_squared):
		self._Cn_squared = Cn_squared
	
	@property
	def outer_scale(self):
		return self._L0

	@outer_scale.setter
	def L0(self, L0):
		self._L0 = L0

		self._recalculate_matrices()