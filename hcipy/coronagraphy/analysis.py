import numpy as np

from ..field import Field, CartesianGrid, PolarGrid, UnstructuredCoords, make_focal_grid
from ..propagation import FraunhoferPropagator
from ..optics import Wavefront
from ..metrics import radial_profile

def CoronagraphAnalysis(object):
	'''A class for performing analysis on a coronagraph.
	
	Parameters
	----------
	coronagraph : OpticalElement
		The optical element that represents the coronagraph. The optical element is
		assumed to propgate from a pre-coronagraphic pupil plane to a post-coronagraphic
		pupil plane.
	pupil : Field
		The pupil of the telescope impinging on the coronagraph.
	pupil_diameter : scalar
		The diameter of the pupil. This is used to calculate the size of a resolution element.
	wavelengths : array_like or scalar
		The wavelengths to use for propagating the images. If a scalar is given, a monochromatic
		propagation will be used.
	'''
	def __init__(self, coronagraph, pupil, pupil_diameter, wavelengths):
		self.coronagraph = coronagraph
		self.pupil = pupil
		self.pupil_diameter = pupil_diameter
		self.wavelengths = wavelengths

	@property
	def wavelengths(self):
		'''The wavelengths to use for the image proapgation.
		'''
		return self._wavelengths

	@wavelengths.setter
	def wavelengths(self, wavelengths):
		if np.isscalar(wavelengths):
			self._wavelengths = np.array([wavelengths])
		else:
			self._wavelengths = np.array(wavelengths)
	
	@property
	def center_wavelength(self):
		'''The center wavelength of the image.
		'''
		return np.mean(self._wavelengths)
	
	@property
	def num_wavelengths(self):
		'''The number of wavelengths used for each image.
		'''
		return len(self._wavelengths)

	def get_performance_offaxis(self, angular_separations, q=64, fov=2):
		'''Get the core throughput and image distortion as function of angular separation.

		Core throughput is defined as the fraction of the planet's light that falls within 
		the >0.5*max(PSF) region of its PSF, normalized to the planet flux incident on the
		primary mirror. This ignores losses from filters, reflections, etc., but includes
		coronagraph mask losses.

		The core throughput will be calculated for all angular separations. If `angular_separations`
		is a one-dimensional array, the angular separation will be taken in the positive 
		x-direction.

		Centroid position is the center of gravity of the planet PSF, taken in 

		Parameters
		----------
		angular_separations : array_like
			The angular separations in lambda_0 / D.
		q : scalar
			The pixels per lambda_0 / D to use for the calculation.
		fov : scalar
			The number of lambda_0 / D radius to look for throughput.
		'''
		# Get region where PSF is larger than half its maximum.
		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, wavelength=self.center_wavelength)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		img = 0
		for wavelength in self.wavelengths:
			wf = Wavefront(self.pupil, wavelength)
			wf.total_power = 1.0 / num_wavelengths
			img += prop(wf).power
		
		reference_region = img > (0.5 * img.max())
		reference_core_throughput = (img * reference_region).sum()

		core_throughputs = []
		centroid_positions = []
		images = []

		# Get core throughput for each angular separation
		for angular_separation in angular_separations:
			img = 0
			if np.isscalar(angular_separation):
				prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid.shifted([-angular_separation, 0]))
			else:
				prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid.shifted([-angular_separation[0], -angular_separation[1]]))
			
			for wavelength in self.wavelengths:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / self.num_wavelengths

				if np.isscalar(angular_separation):
					wf.electric_field *= np.exp(2j * np.pi * self.center_wavelength / (wavelength * self.pupil_diameter) * self.pupil.grid.x * angular_separation)
				else:
					wf.electric_field *= np.exp(2j * np.pi * self.center_wavelength / wavelength / self.pupil_diameter * np.dot(self.pupil.grid.coords, angular_separation))
				
				img += prop(self.coronagraph(wf)).power
			
			img = Field(img, normalized_focal_grid)

			core_throughputs.append((img * reference_region).sum())
			centroid_positions.append() # FIXME
			images.append(Field(img, focal_grid))
		
		output = {
			'angular_separations': angular_separations,
			'core_throughputs': core_throughputs,
			'centroid_positions': centroid_positions,
			'core_throughput_reference': reference_core_throughput,
			'region': reference_region,
			'images': images}

		return output

	def make_image_with_pointings(self, prop, pointings):
		'''Make an image with a number of pointing positions.

		Parameters
		----------
		prop : Propagator
			The propagator from the pupil to the focal plane. The image is returned on
			this focal plane.
		pointings : Grid

		'''
		img = 0

		for wavelength in self.wavelengths:
			for p in pointings.as_('cartesian').points:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / (self.num_wavelengths * len(pointings))

				wf.electric_field *= np.exp(2j * np.pi * self.center_wavelength / wavelength / self.pupil_diameter * np.dot(self.pupil.grid.as_('cartesian').coords, p))

				img += prop(self.coronagraph(wf)).power
		
		return img

	def get_performance_stellar_diameter(self, stellar_diameters, num_samples=100, q=16, fov=32):
		'''Get the image and profile for various stellar diameters.

		This function uses `num_samples` propagations through the coronagraph to simulate
		an image from a finite size star.

		Parameters
		----------
		stellar_diameters : array_like
			The stellar diameters for which to calculate the images and profiles.
		num_samples : int
			The number of samples to take to simulate the finite size star.
		q : scalar
			The number of pixels per resolution element.
		num_airy : scalar
			The number of resolution elements visible in the images.
		
		Returns
		-------
		dictionary
			All images, profiles, and supporting information.
		'''
		images = []
		profiles = []
		profiles_std = []
		profiles_num = []

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, wavelength=self.center_wavelength)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		r = np.sqrt(np.random.uniform(0, 1, num_samples)) / 2
		theta = np.random.uniform(0, 2 * np.pi, num_samples)
		samples = PolarGrid(UnstructuredCoords((r, theta)))

		for stellar_diameter in stellar_diameters:
			img = self.make_image_with_pointings(prop, samples.scaled(stellar_diameter))
			img.grid = normalized_focal_grid
		
			r, y, y_std, y_num = radial_profile(img, 1.0 / q)

			images.append(img)
			angular_separations = r
			profiles.append(y)
			profiles_std.append(y_std)
			profiles_num.append(y_num)

		output = {
			'images': images,
			'profiles': profiles,
			'stellar_diameters': stellar_diameter,
			'angular_separations': angular_separations,
			'profiles_std': profiles_std,
			'profiles_num': profiles_num}

		return output

	def get_performance_pointing_jitter(self, pointing_jitters, num_samples=100, q=4, fov=32):
		'''Get the image and profile for various values for a tip-tilt jitter (in both axes).

		Parameters
		----------
		pointing_jitters : array_like
			The pointing jitters in resolution elements at the center wavelength.
		num_samples : int
			The number of samples to take to simulate the pointing jitter.
		q : scalar
			The number of pixels per resolution element.
		num_airy : scalar
			The number of resolution elements visible in the images.
		
		Returns
		-------
		dictionary
			All images, profiles, and supporting information.
		'''
		images = []
		profiles = []
		profiles_std = []
		profiles_num = []

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, wavelength=self.center_wavelength)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		x = np.random.randn(num_samples) / np.sqrt(2)
		y = np.random.randn(num_samples) / np.sqrt(2)

		samples = CartesianGrid(UnstructuredCoords((x, y)))

		for pointing_jitter in pointing_jitters:
			img = self.make_image_with_pointings(prop, samples.scaled(pointing_jitter))
			img.grid = normalized_focal_grid

			r, y, y_std, y_num = radial_profile(img, 1.0 / q)

			images.append(img)
			angular_separations = r
			profiles.append(y)
			profiles_std.append(y_std)
			profiles_num.append(y_num)

		output = {
			'images': images,
			'profiles': profiles,
			'pointing_jitters': pointing_jitters,
			'angular_separations': angular_separations,
			'profiles_std': profiles_std,
			'profiles_num': profiles_num}

		return output

	def make_image_with_aberration(self, prop, aberration=0, coeffs=0):
		img = 0

		if np.isscalar(coeffs):
			coeffs = [coeffs]

		for wavelength in self.wavelengths:
			for c in coeffs:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / (self.num_wavelengths * len(coeffs))

				wf.electric_field *= np.exp(1j * c * aberration * coeffs * self.center_wavelength / wavelength)

				img += prop(self.coronagraph(wf)).power
		
		return img

	def get_low_order_sensitivity_drift(self, modes, rms_drift, num_samples=100, q=4, fov=32):
		'''Get the image and profile for drifting aberrations on several modes.

		Parameters
		----------
		modes : ModeBasis
			The modes for which to calculate the images and profiles. All modes are assumed
			to be normalized to a RMS of 1.
		rms_drift : scalar
			The rms wavefront drift in meters for the modes.
		num_samples : int
			The number of samples to take to simulate the finite size star.
		q : scalar
			The number of pixels per resolution element.
		num_airy : scalar
			The number of resolution elements visible in the images.
		
		Returns
		-------
		dictionary
			All images, profiles, and supporting information.
		'''
		images = []
		profiles = []
		profiles_std = []
		profiles_num = []

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, wavelength=self.center_wavelength)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)
		reference_image = self.make_image_with_aberration(prop)

		coeffs = np.random.randn(num_samples)

		for mode in modes:
			img = make_image_with_aberration(prop, mode, coeffs * rms_drift * 2 * np.pi / self.center_wavelength)

			r, y, y_std, y_num = radial_profile(img, 1.0 / q)

			images.append(img)
			angular_separations = r
			profiles.append(y)
			profiles_std.append(y_std)
			profiles_num.append(y_num)
		
		output = {
			'images': images,
			'profiles': profiles,
			'rms_drift': rms_drift,
			'angular_separations': angular_separations,
			'profiles_std': profiles_std,
			'profiles_num': profiles_num}

		return output
	
	def get_low_order_sensitivity_jitter(self, modes, rms_jitter, num_samples=100, q=4, fov=32):
		'''Get the image and profile for jittering aberrations on several modes.

		Parameters
		----------
		modes : ModeBasis
			The modes for which to calculate the images and profiles. All modes are assumed
			to be normalized to a RMS of 1.
		rms_jitter : scalar
			The wavefront jitter in meters for the modes.
		num_samples : int
			The rms number of samples to take to simulate the finite size star.
		q : scalar
			The number of pixels per resolution element.
		num_airy : scalar
			The number of resolution elements visible in the images.
		
		Returns
		-------
		dictionary
			All images, profiles, and supporting information.
		'''
		images = []
		profiles = []
		profiles_std = []
		profiles_num = []

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, wavelength=self.center_wavelength)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		coeffs = np.random.randn(num_samples)

		for mode in modes:
			img = make_image_with_aberration(prop, mode, coeffs * rms_jitter * 2 * np.pi / self.center_wavelength)
			img.grid = normalized_focal_grid

			r, y, y_std, y_num = radial_profile(img, 1.0 / q)

			images.append(img)
			angular_separations = r
			profiles.append(y)
			profiles_std.append(y_std)
			profiles_num.append(y_num)

		output = {
			'images': images,
			'profiles': profiles,
			'rms_jitter': rms_jitter,
			'angular_separations': angular_separations,
			'profiles_std': profiles_std,
			'profiles_num': profiles_num}

		return output
