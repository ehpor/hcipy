import numpy as np
from .apodization import SurfaceApodizer
from .optical_element import OpticalElement
from ..field import Field

def closest_points(point_grid, evaluated_grid):
	from scipy import spatial

	tree = spatial.KDTree(point_grid.points)
	d, i = tree.query(evaluated_grid.points)

	return Field(i, evaluated_grid), Field(d, evaluated_grid)

class MicroLensArray(OpticalElement):
	def __init__(self, input_grid, lenslet_grid, focal_length, lenslet_shape=None):
		self.input_grid = input_grid
		self.focal_length = focal_length
		
		self.mla_grid = lenslet_grid
		
		if lenslet_shape is None:
			indices, distances = closest_points(lenslet_grid, input_grid)

			self.mla_index = indices
			self.mla_opd = (-1 / (2*focal_length)) * distances**2
		else:
			self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
			self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)

			for i, (x,y) in enumerate(lenslet_grid.as_('cartesian').points):
				shifted_grid = input_grid.shifted((x,y))
				mask = lenslet_shape(shifted_grid) != 0
				
				self.mla_opd[mask] = (-1 / (2*focal_length)) * (shifted_grid.x[mask]**2 + shifted_grid.y[mask]**2)
				self.mla_index[mask] = i
			
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * self.mla_opd * 2*np.pi / wf.wavelength)
		return wf
		
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field /= np.exp(1j * self.mla_opd * 2*np.pi / wf.wavelength)
		return wf

def aspheric_surface(radius_of_curvature, refractive_index, conic_constant=0, aspheric_coefficients=[]):

	def func(grid):
		x = grid.x
		y = grid.y
		r = np.hypot(x, y)

		# Start with a conic surface
		curvature = 1/radius_of_curvature
		alpha = (1+conic_constant) * curvature**2 * r**2
		opd = -r**2/(radius_of_curvature * (1 + np.sqrt(1-alpha)))

		# Add aspheric coefficients
		# Only use the even modes and start at 4, because 0 is piston and 2 is the conic surface
		for ai, coef in enumerate(aspheric_coefficients):
			power_index = 4 + ai * 2
			opd += -coef * (r/radius_of_curvature)**power_index
		return SurfaceApodizer(opd, refractive_index)
	return func

class EvenAsphereMicroLensArray(OpticalElement):
	def __init__(self, input_grid, lenslet_grid, radius_of_curvature, wavelength=1, refractive_index=1.5, conic_constant=0, aspheric_coefficients=[], lenslet_shape=None):
		self.input_grid = input_grid
		
		self.mla_grid = lenslet_grid
		self.n = refractive_index
		self.radius_of_curvature = radius_of_curvature
		self.conic_constant = conic_constant

		#self.focal_length = radius_of_curvature/(refractive_index-1)

		self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
		self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)
		self.mla_amp = Field(np.zeros(self.input_grid.size), self.input_grid)
		self.mla_cin = Field(np.zeros(self.input_grid.size), self.input_grid)

		for i, (x,y) in enumerate(lenslet_grid.as_('cartesian').points):
			shifted_grid = input_grid.shifted((x,y))
			mask = lenslet_shape(shifted_grid) != 0
			
			if np.count_nonzero(mask) > 0:
				xs = shifted_grid.x[mask]
				ys = shifted_grid.y[mask]
				r = np.hypot(xs, ys)

				# Start with a conic surface
				curvature = 1/self.radius_of_curvature
				alpha = 1 - (1+self.conic_constant) * curvature**2 * r**2
				self.mla_opd[mask] = curvature * r**2 /(1+np.sqrt(alpha))

				# Add aspheric coefficients
				# Only use the even modes and start at 4, because 0 is piston and 2 is the conic surface
				for ai, coef in enumerate(aspheric_coefficients):
					power_index = 4 + ai * 2
					self.mla_opd[mask] += -coef * (r/r.max())**power_index

				self.mla_index[mask] = i
		self.mla_opd = -self.mla_opd
				
		self.surface = SurfaceApodizer(self.mla_opd, refractive_index)

	def forward(self, wavefront):
		return self.surface.forward(wavefront)
		
	def backward(self, wavefront):
		return self.surface.backward(wavefront)