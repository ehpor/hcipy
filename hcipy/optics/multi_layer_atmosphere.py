import numpy as np
from .optical_element import OpticalElement

def random_velocity(velocity):
	"""
	Generate a random 2D vector with the specified magnitude.
	"""
	if np.isscalar(velocity):
		#~ return np.random.randn(2) * velocity
		theta = np.random.rand() * 2*np.pi
		return velocity * np.array([np.cos(theta), np.sin(theta)])
	return velocity

class AtmosphericModel(OpticalElement):
	"""
	Modeling abberations for atmospheric turbulence using discrete atmospheric layers.
	If r_0 is given, the C_n^2 profile is scaled to that Fried parameter.
	Turbulence_layers is a list of tuples: (height, velocity, C_n^2).
	If a velocity is scalar, the direction for that layer is chosen randomly.
	"""
	def __init__(self, spectral_noise_factory, turbulence_layers, zenith_angle=0, off_axis_angle=0, include_scintilation=False):
		self.phase_screen_factory = spectral_noise_factory
		
		self.include_scintilation = include_scintilation
		self.zenith_angle = zenith_angle

		self.heights, self.velocities, self.Cn_squared = zip(*turbulence_layers)
		self.velocities = [random_velocity(v) for v in self.velocities]

		self.t = None
		self.off_axis_angle = off_axis_angle

		# Sort by altitude from high to low
		inds = np.argsort(self.heights)[::-1]
		self.heights = np.array([self.heights[i] for i in inds], dtype='float')
		self.velocities = [self.velocities[i] for i in inds]
		self.Cn_squared = np.array([self.Cn_squared[i] for i in inds], dtype='float')
	
	@property
	def t(self):
		return self._t
	
	@t.setter
	def t(self, t):
		if t is None:
			self.phase_screens = [self.phase_screen_factory.make_random() for i in len(turbulence_layers)]
			self._t = 0
		else:
			self._t = t

	def evolve_until(self, t=None):
		self.t = t
	
	def forward(self, wavefront):
		k = 2*np.pi / wavefront.wavelength

		if self.include_scintilation:
			wf = wavefront
			for i, (height, velocity, Cn, phase_screen) in enumerate(self.layers):
				# TODO: store propagators once
				shift = self.t * velocity - np.sin(self.off_axis_angle * height / np.cos(self.zenith_angle)
				phase_screen = phase_screen.shifted(shift) * np.sqrt(Cn * k**2 * 0.423)
				phase_screen = phase_screen()

				wf.electric_field *= np.exp(1j * phase_screen)

				if i+1 >= len(self.layers):
					next_height = 0
				else:
					next_height = self.heights[i+1]
				z = (height - next_height) / np.cos(self.zenith_angle)
				prop = FresnelPropagator(wavefront.electric_field.grid, z)
				wf = prop(wf)
		else:
			cumulated_phase_screen = self.phase_screen_factory.make_zero()

			for height, velocity, Cn, phase_screen in self.layers:
				shift = self.t * velocity - np.sin(self.off_axis_angle) * height / np.cos(self.zenith_angle)
				phase_screen = phase_screen.shifted(shift) * np.sqrt(Cn * k**2 * 0.423)
				cumulated_phase_screen += phase_screen
			cumulated_phase_screen = cumulated_phase_screen()

			wf = wavefront.copy()
			wf.electric_field *= np.exp(1j * cumulated_phase_screen)
		return wf

	@property
	def layers(self):
		return zip(self.heights, self.velocities, self.Cn_squared, self.phase_screens)

def kolmogorov_psd(grid):
	res = np.zeros()
	return 0.023 * (u/(2*np.pi))**(-11/3.)

def von_karman_psd(grid, u_o=0.1):
	return 0.0299 * ((u**2 + u_o**2)/(2*np.pi)**2)**(-11/6.)

def modified_von_karman_psd(grid, u_o=0.1, u_i=100):
	return VonKarmanPSD(u, u_o) * np.exp(-u**2/u_i**2)

def make_standard_multilayer_atmosphere(fried_parameter=1, wavelength=500e-9, zenith_angle=0):
	"""
	Standard atmosphere from Guyon (2005):
	Limits of Adaptive Optics for high contrast imaging.
	Returns an unnormalized Cn_squared profile.
	"""
	heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 10, 10, 10, 10])
	Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350])
	Cn_squared = scale_Cn_squared_to_fried_parameter(Cn_Squared, fried_parameter, wavelength, zenith_angle)
	
	return (heights, velocities, Cn_squared)

def scale_Cn_squared_to_fried_parameter(Cn_squared, fried_parameter, wavelength=500e-9, zenith_angle=0):
	k = 2*np.pi / wavelength
	Cn_squared /= np.sum(Cn_squared)
	Cn_squared *= fried_parameter**(-5/3.) / (0.423 * k**2 / np.cos(zenith_angle))
	return Cn_squared

def get_fried_parameter(Cn_squared, wavelength=500e-9, zenith_angle=0):
	k = 2*np.pi / wavelength
	return (0.423 * k**2 * np.sum(Cn_squared) / np.cos(zenith_angle))**(-3/5.)