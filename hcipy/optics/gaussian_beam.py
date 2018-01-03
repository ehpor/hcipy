import numpy as np

class GaussianBeam(object):
	def __init__(self, w0, z, wavelength):
		self.w0 = w0
		self.z = z
		self.wavelength = wavelength
	
	@property
	def beam_waist(self):
		return self.w0
	
	@beam_waist.setter
	def beam_waist(self, w0):
		self.w0 = w0

	@property
	def zR(self):
		return np.pi * self.w0**2 / self.wavelength
	
	@zR.setter
	def zR(self, zR):
		self.w0 = np.sqrt(zR * self.wavelength / np.pi)

	rayleigh_distance = zR

	@property
	def q(self):
		return self.z + 1j * self.zR
	
	@q.setter
	def q(self, q):
		self.z = np.real(q)
		self.zR = np.imag(q)

	complex_beam_parameter = q

	@property
	def theta(self):
		return self.wavelength / (np.pi * self.w0)
	
	@theta.setter
	def theta(self, theta):
		self.w0 = self.wavelength / (theta * np.pi)
	
	beam_divergence = theta

	@property
	def R(self):
		return self.z * (1 + (self.zR / self.z)**2)
	
	radius_of_curvature = R

	@property
	def psi(self):
		return np.arctan(self.z / self.zR)
	
	gouy_phase = psi

	@property
	def w(self):
		return self.w0 * np.sqrt(1 + (self.z / self.zR)**2)

	beam_radius = w

	@property
	def FWHM(self):
		return self.w * np.sqrt(2 * np.log(2))
	
	full_width_half_maximum = FWHM

	@property
	def k(self):
		return 2 * np.pi / self.wavelength
	
	@k.setter
	def k(self, k):
		self.wavelength = 2 * np.pi / k
	
	wave_number = k
	
	def evaluate(self, grid):
		if grid.is_('cartesian'):
			r2 = grid.x**2 + grid.y**2
		else:
			r2 = grid.as_('polar').r**2
		
		K1 = self.w0 / self.w
		K2 = np.exp(-r2 / self.w**2)
		K3 = np.exp(-1j * (self.k * self.z + self.k * r2 / (2 * self.R) - self.psi))

		return Field(K1 * K2 * K3, grid)
	
	__call__ = evaluate