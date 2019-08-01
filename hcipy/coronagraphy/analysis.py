import numpy as np

from ..field import Field, CartesianGrid, PolarGrid, UnstructuredCoords, make_focal_grid
from ..propagation import FraunhoferPropagator
from ..optics import Wavefront
from ..metrics import radial_profile

class CoronagraphAnalysis(object):
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
	def __init__(self, coronagraph, non_coronagraph, pupil, pupil_diameter, wavelengths):
		self.coronagraph = coronagraph
		self.non_coronagraph = non_coronagraph
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
		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, reference_wavelength=self.center_wavelength, focal_length=1)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		ref_img = self.make_image_with_pointings(prop, coronagraphic=False)

		reference_region = ref_img > (0.5 * ref_img.max())
		reference_core_throughput = (ref_img * reference_region).sum()

		core_throughputs = []
		centroid_positions = []
		images = []

		# Get core throughput for each angular separation
		for angular_separation in angular_separations:
			if np.isscalar(angular_separation):
				prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid.shifted([angular_separation, 0]))
				pointing = CartesianGrid(UnstructuredCoords([np.array([angular_separation]), np.array([0])]))
			else:
				prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid.shifted([angular_separation[0], angular_separation[1]]))
				pointing = CartesianGrid(UnstructuredCoords([np.array([angular_separation[0]]), np.array([angular_separation[1]])]))

			img = self.make_image_with_pointings(prop, pointing)
			img = Field(img, normalized_focal_grid)

			core_throughputs.append((img * reference_region).sum())

			centroid_x = ((img * reference_region) * img.grid.x).sum() / (img * reference_region).sum()
			centroid_y = ((img * reference_region) * img.grid.y).sum() / (img * reference_region).sum()
			centroid_positions.append(np.array([centroid_x, centroid_y]))

			images.append(Field(img / ref_img.max(), focal_grid))
			print(img.sum())

		output = {
			'angular_separations': angular_separations,
			'core_throughputs': core_throughputs,
			'centroid_positions': centroid_positions,
			'core_throughput_reference': reference_core_throughput,
			'region': reference_region,
			'images': images}

		return output

	def make_image_with_pointings(self, prop, pointings=None, coronagraphic=True):
		'''Make an image with a number of pointing positions.

		Parameters
		----------
		prop : Propagator
			The propagator from the pupil to the focal plane. The image is returned on
			this focal plane.
		pointings : Grid

		'''
		img = 0

		if pointings is None:
			pointings = CartesianGrid(UnstructuredCoords([np.array([0]), np.array([0])]))

		for wavelength in self.wavelengths:
			for p in pointings.as_('cartesian').points:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / (self.num_wavelengths * len(pointings))

				wf.electric_field *= np.exp(2j * np.pi * self.center_wavelength / wavelength / self.pupil_diameter * np.dot(p, self.pupil.grid.as_('cartesian').coords))

				if coronagraphic:
					img += prop(self.coronagraph(wf)).power
				else:
					img += prop(self.non_coronagraph(wf)).power
		
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

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, reference_wavelength=self.center_wavelength, focal_length=1)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		ref_img = self.make_image_with_pointings(prop, coronagraphic=False)

		normalized_focal_grid = make_focal_grid(q, fov)

		r = np.sqrt(np.random.uniform(0, 1, num_samples)) / 2
		theta = np.random.uniform(0, 2 * np.pi, num_samples)
		samples = PolarGrid(UnstructuredCoords((r, theta)))

		for stellar_diameter in stellar_diameters:
			print(stellar_diameter)
			img = self.make_image_with_pointings(prop, samples.scaled(stellar_diameter))

			img.grid = normalized_focal_grid
			img /= ref_img.max()
		
			r, y, y_std, y_num = radial_profile(img, 2.0 / q)

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

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, reference_wavelength=self.center_wavelength, focal_length=1)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		ref_img = self.make_image_with_pointings(prop, coronagraphic=False)

		x = np.random.randn(num_samples) / np.sqrt(2)
		y = np.random.randn(num_samples) / np.sqrt(2)

		samples = CartesianGrid(UnstructuredCoords((x, y)))

		for pointing_jitter in pointing_jitters:
			img = self.make_image_with_pointings(prop, samples.scaled(pointing_jitter))

			img.grid = normalized_focal_grid
			img /= ref_img.max()

			r, y, y_std, y_num = radial_profile(img, 2.0 / q)

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

	def make_images_with_aberration(self, prop, aberration=0, coeffs=0):
		if np.isscalar(coeffs):
			coeffs = [coeffs]

		images = []
		for c in coeffs:
			img = 0
			for wavelength in self.wavelengths:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / (self.num_wavelengths * len(coeffs))

				wf.electric_field *= np.exp(1j * c * aberration * self.center_wavelength / wavelength)

				img += prop(self.coronagraph(wf)).power
			images.append(img)
		
		return images
	
	def make_electric_fields_with_aberration(self, prop, aberration=0, coeffs=0):
		if np.isscalar(coeffs):
			coeffs = [coeffs]

		electric_fields = []
		for c in coeffs:
			electric_field = []
			for wavelength in self.wavelengths:
				wf = Wavefront(self.pupil, wavelength)
				wf.total_power = 1.0 / (self.num_wavelengths * len(coeffs))

				wf.electric_field *= np.exp(1j * c * aberration * self.center_wavelength / wavelength)

				img += prop(self.coronagraph(wf)).electric_field
			electric_fields.append(electric_field)
		
		return electric_fields

	def get_low_order_sensitivity_drift(self, modes, rms_drift, q=4, fov=32):
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

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, reference_wavelength=self.center_wavelength, focal_length=1)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		ref_img = self.make_image_with_pointings(prop, coronagraphic=False)

		reference_image = self.make_images_with_aberration(prop)[0]
		reference_image /= ref_img.max()

		for mode in modes:
			print(mode)
			img = self.make_images_with_aberration(prop, mode, rms_drift * 2 * np.pi / self.center_wavelength)[0]
			img = Field(img, normalized_focal_grid)
			img /= ref_img.max()

			img -= reference_image

			r, y, y_std, y_num = radial_profile(img, 2.0 / q)

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

		focal_grid = make_focal_grid(q, fov, pupil_diameter=self.pupil_diameter, reference_wavelength=self.center_wavelength, focal_length=1)
		prop = FraunhoferPropagator(self.coronagraph.output_grid, focal_grid)

		normalized_focal_grid = make_focal_grid(q, fov)

		ref_img = self.make_image_with_pointings(prop, coronagraphic=False)
		ref_noncoronagraphic = self.make_electric_fields_with_aberration(prop, coronagraphic=False)
		ref_noncoronagraphic_norm = np.array(ref_noncoronagraphic)
		ref_coronagraphic = self.make_electric_fields_with_aberration(prop, coronagraphic=False)
		ref_coronagraphic = np.array(ref_coronagraphic)

		coeffs = np.random.randn(num_samples)

		for mode in modes:
			print(mode)
			imgs = self.make_images_with_aberration(prop, mode, coeffs * rms_jitter * 2 * np.pi / self.center_wavelength)
			img = np.std(imgs, axis=0)

			img = Field(img, normalized_focal_grid)
			img /= ref_img.max()

			r, y, y_std, y_num = radial_profile(img, 2.0 / q)

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
