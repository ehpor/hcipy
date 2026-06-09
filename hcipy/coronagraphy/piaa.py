def find_optimal_apodization(aperture, focal_plane_mask, num_iterations, wavelengths, aperture_mask=None, final_mask=None):
	'''
	'''
	
	if aperture_mask is None:
		aperture_mask = 1.0 * (aperture.copy()>0)
	
	if final_mask is None:
		final_mask = 1.0 * (final_mask.copy()>0)

	grid = aperture.grid
	focal_grid = focal_plane_mask.grid
	propagator = FraunhoferPropagator(grid, focal_grid)

	fpm_apodizer = Apodizer(focal_plane_mask)

	for i in range(num_iterations):

		lyot_mode = grid.zeros()
		for wave in wavelengths:
			wf = Wavefront(aperture * aperture_mask, wave)
			wf.total_power = 1

			wf_m1 = propagator(wf)
			temp = propagator.backward( fpm_apodizer(wf_m1) ).electric_field
			lyot_mode += abs(temp)

		aperture = lyot_mode / wavelengths.size

	return aperture * final_mask

def swap_fields(x, y):
	temp = y.copy()
	y = x.copy()
	x = temp.copy()
	
	return x, y

def calculate_encircled_energy(field, bin_size, normalize=True):
	r, rad_profile, prof_std, ncount = radial_profile(field, bin_size, statistic='mean')

	encircled_energy = integrate.cumulative_trapezoid(rad_profile * r, r, initial=0)
	
	if normalize:
		encircled_energy /= encircled_energy.max()

	return r, encircled_energy

class RadialPiaaSag():

	def __init__(self, input_distribution, output_distribution):
		
		bin_size = input_distribution.grid.delta[0]
		self._rin, self._input_profile = calculate_encircled_energy(input_distribution, bin_size, normalize=True)
		self._rout, self._output_profile = calculate_encircled_energy(output_distribution, bin_size, normalize=True)
		
		# Find the edges of the power distributions
		self._input_inner_edge = self._rin[np.argmin(abs(self._input_profile))]
		self._input_outer_edge = self._rin[np.argmin(abs(self._input_profile - 1))]
		
		self._output_inner_edge = self._rin[np.argmin(abs(self._output_profile))]
		self._output_outer_edge = self._rout[np.argmin(abs(self._output_profile - 1))]

	def get_radial_points(self):
		return self._rin.copy()

	def forward(self, input_radius):
		mask = (input_radius >= 0) * (input_radius <= self._input_outer_edge)

		output_radius = np.zeros_like(input_radius)
		new_C1 = np.interp(input_radius[mask], self._rin, self._input_profile)
		output_radius[mask] = np.interp(new_C1, self._output_profile, self._rin)
		output_radius[input_radius > self._input_outer_edge] = input_radius[input_radius > self._input_outer_edge]

		return output_radius

	def backward(self, output_radius):
		mask = (output_radius >= 0) * (output_radius <= self._output_outer_edge)

		input_radius = np.zeros_like(output_radius)
		new_C2 = np.interp(output_radius[mask], self._rout, self._output_profile)
		input_radius[mask] = np.interp(new_C2, self._input_profile, self._rout)
		input_radius[output_radius > self._output_outer_edge] = output_radius[output_radius > self._output_outer_edge]

		return input_radius

	def create_piaa_surface_sag(self, radial_points, grid, distance, refractive_index=1.5):
		
		# Make the surface sags
		tan_alpha = np.nan_to_num( ((refractive_index**2 - 1) + (distance * (refractive_index - 1))**2 / (self.forward(radial_points) - radial_points)**2)**(-1/2) )
		surface_sag_1 = integrate.cumulative_trapezoid(tan_alpha, radial_points, initial=0)
		surface_sag_1 -= np.max(surface_sag_1)
		
		# Surface 1 needs to be negative.
		# TODO: figure out the correct boundary conditions for the integral.
		surface_1 = Field(np.interp(grid.as_('polar').r, radial_points, -surface_sag_1), grid)

		# Make surface 2
		tan_beta = np.nan_to_num( ((refractive_index**2 - 1) + (distance * (refractive_index - 1))**2 / (self.backward(radial_points) - radial_points)**2)**(-1/2) )
		surface_sag_2 = integrate.cumulative_trapezoid(tan_beta, radial_points, initial=0)
		surface_sag_2 -= np.max(surface_sag_2)
		surface_2 = Field(np.interp(grid.as_('polar').r, radial_points, surface_sag_2), grid)

		return surface_1, surface_2