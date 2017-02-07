import numpy as np
from .optical_element import OpticalElement

class AtmosphericModel(OpticalElement):
	"""
	Modeling abberations for atmospheric turbulence using discrete atmospheric layers.
	If r_0 is given, the C_n^2 profile is scaled to that Fried parameter.
	Turbulence_layers is a list of tuples: (height, velocity, C_n^2).
	If a velocity is scalar, the direction for that layer is chosen randomly.
	"""
	def __init__(self, spectral_noise_factory, turbulence_layers, r_0=None, zenith_angle=0, include_scintilation=False, remove_modes=None):
		self.spectral_noise_factory = spectral_noise_factory
		
		self.include_scintilation = include_scintilation
		self.zenith_angle = zenith_angle

		if not remove_modes is None:
			self.remove_modes = True
			self.modes_transformation = remove_modes.

class AtmosphericModel(OpticalElement):
	"""
	Modeling abberations for atmospheric turbulence using discrete atmospheric layers.
	If r_0 is given, the C_n^2 profile is scaled to that Fried parameter.
	Turbulence_layers is a tuple: (heights, velocities, C_n^2).
	If a velocity is scalar, a random direction is chosen for that layer.
	"""
	def __init__(self, phase_screen_factory, turbulence_layers, wavelength=500e-9, r_0=None, zenith_angle=0, fresnel_propagation=False, remove_modes=None):
		self.phase_screen_factory = phase_screen_factory

		self.fresnel_propagation = fresnel_propagation
		self.zenith_angle = zenith_angle
		self.wavelength = wavelength
		
		if not remove_modes is None:
			self.remove_modes = True
			self.modes_transformation = remove_modes.get_linear_transformation()
			self.modes_transformation_inv = self.modes_transformation.inverse_cutoff(1e-10)
		else:
			self.remove_modes = False

		self.heights, self.velocities, self.Cn_squared = turbulence_layers
		self.velocities = [random_velocity(v) for v in self.velocities]
		self.phase_screens = [self.phase_screen_factory.get_random_phase_screen() for h in self.heights]
		
		# Sort by altitude; highest first.
		inds = np.argsort(self.heights)[::-1]
		self.heights = np.array([self.heights[i] for i in inds], dtype='float')
		self.velocities = [self.velocities[i] for i in inds]
		self.Cn_squared = np.array([self.Cn_squared[i] for i in inds], dtype='float')

		if not r_0 is None:
			self.fried_parameter = r_0
	
	def get_wavefront(self, t=None, theta=0, wavelength=None):
		"""
		Generate a phase screen from the initial parameters for
		t=time, theta=angle from on-axis, wavelength.
		If t is None, then a new, independent wavefront is generated and returned.
		"""
		if wavelength is None:
			wavelength = self.wavelength
		k = 2*np.pi / wavelength
		
		if t is None:
			self.phase_screens = [self.phase_screen_factory.get_random_phase_screen() for h in self.heights]
			t = 0
		
		if self.fresnel_propagation:
			electric_field = np.exp(1j * self.phase_screen_factory.get_zero_phase_screen()())
			electric_field = SampledFunction(electric_field, self.phase_screen_factory.output_grid)
			wavefront = Wavefront(electric_field, wavelength)
			
			for i, (height, velocity, Cn, phase_screen) in enumerate(self.layers):
				shift = t * velocity - np.sin(theta) * height / np.cos(self.zenith_angle)
				phase_screen = phase_screen.shift(shift) * np.sqrt(Cn * k**2)
				phase_screen = phase_screen()
				
				if self.remove_modes:
					phase_screen -= self.modes_transformation(self.modes_transformation_inv(phase_screen))
				
				wavefront.electric_field *= np.exp(1j * phase_screen)
				
				if i+1 >= len(self.layers):
					next_height = 0
				else:
					next_height = self.heights[i+1]
				z = (height - next_height) / np.cos(self.zenith_angle)
				wavefront = propagate_paraxial(wavefront, z)
		else:
			cumulated_phase_screen = self.phase_screen_factory.get_zero_phase_screen()
			
			for height, velocity, Cn, phase_screen in self.layers:
				shift = t * velocity - np.sin(theta) * height / np.cos(self.zenith_angle)
				phase_screen = phase_screen.shift(shift) * np.sqrt(Cn * k**2 * 0.423)
				cumulated_phase_screen += phase_screen
			cumulated_phase_screen = cumulated_phase_screen()
			
			if self.remove_modes:
				cumulated_phase_screen -= self.modes_transformation(self.modes_transformation_inv(cumulated_phase_screen))
			
			electric_field = np.exp(1j * cumulated_phase_screen)
			electric_field = SampledFunction(electric_field, self.phase_screen_factory.output_grid)
			wavefront = Wavefront(electric_field, wavelength)
		
		return wavefront

	@property
	def layers(self):
		return zip(self.heights, self.velocities, self.Cn_squared, self.phase_screens)

	@property
	def wavelength(self):
		return 2*np.pi / self.k

	@wavelength.setter
	def wavelength(self, wavelength):
		self.k = 2*np.pi / wavelength

	@property
	def fried_parameter(self):
		return (0.423 * self.k**2 * np.sum(self.Cn_squared) / np.cos(self.zenith_angle))**(-3/5.)

	@fried_parameter.setter
	def fried_parameter(self, r_0):
		self.Cn_squared /= np.sum(self.Cn_squared)
		self.Cn_squared *= r_0**(-5/3.) / (0.423 * self.k**2 / np.cos(self.zenith_angle))
	
	@property
	def output_grid(self):
		return self.phase_screen_factory.output_grid

def KolmogorovPSD(grid):
	res = np.zeros()
	return 0.023 * (u/(2*np.pi))**(-11/3.)

def VonKarmanPSD(grid, u_o=0.1):
	return 0.0299 * ((u**2 + u_o**2)/(2*np.pi)**2)**(-11/6.)

def ModifiedVonKarmanPSD(grid, u_o=0.1, u_i=100):
	return VonKarmanPSD(u, u_o) * np.exp(-u**2/u_i**2)

def StandardMultilayerAtmosphere():
	"""
	Standard atmosphere from Guyon (2005):
	Limits of Adaptive Optics for high contrast imaging.
	Returns an unnormalized Cn_squared profile.
	"""
	heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
	velocities = np.array([10, 10, 10, 10, 10, 10])
	Cn_squared = np.array([0.2283, 0.0883, 0.0666, 0.1458, 0.3350, 0.1350])
	return (heights, velocities, Cn_squared)