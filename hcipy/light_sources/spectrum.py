import numpy as np
from scipy import interpolate
from astropy.io import fits
from scipy.stats import binned_statistic
import time
from ..config import Configuration
import os

import matplotlib.pyplot as plt

speed_of_light = 299792.458 # km / s

class Spectrum(object):
	def redshifted(self, velocity=None, redshift=None):
		if redshift is None:
			if velocity is None:
				raise ValueError('Either velocity or redshift needs to be supplied.')
			else:
				redshift = np.sqrt(1 + velocity / speed_of_light) / np.sqrt(1 - velocity / speed_of_light) - 1
		return self.get_redshifted_spectrum(redshift)
	
	def get_redshifted_spectrum(self, redshift):
		raise NotImplementedError()

class AnalyticalSpectrum(Spectrum):
	def approximate(self, num_wavelengths, min_wavelength, max_wavelength):
		raise NotImplementedError()

class TabulatedSpectrum(Spectrum):
	def __init__(self, spectrum):
		self._spectrum = spectrum
	
	def get_redshifted_spectrum(self, redshift):
		spectrum = self._spectrum.copy()
		spectrum.grid.scale(1 + redshift)

		return TabulatedSpectrum(spectrum)
	
	def downsample(self, new_wavelengths):
		pass

class BlackBodySpectrum(AnalyticalSpectrum):
	def __init__(self, temperature, angular_diameter, redshift=0):
		self.temperature = temperature
		self.angular_diameter = angular_diameter
		self._redshift = redshift
	
	@property
	def angular_diameter(self):
		return 2 * 360 * 3600 * np.sqrt(self.solid_angle / np.pi) / (2 * np.pi)
	
	@angular_diameter.setter
	def angular_diameter(self, angular_diameter):
		self.solid_angle = np.pi * (2*np.pi * angular_diameter / (2 * 360 * 3600)**2)

	def get_redshifted_spectrum(self, redshift):
		new_redshift = (1 + self._redshift) * (1 + redshift) - 1
		return BlackBodySpectrum(self.temperature, self.angular_diameter, new_redshift)

	def approximate(self, num_wavelengths, min_wavelength, max_wavelength):
		raise NotImplementedError()

class GaussianSpectrum(AnalyticalSpectrum):
	def __init__(self, center_wavelength, spectral_width):
		self.center_wavelength = center_wavelength
		self.spectral_width = spectral_width
	
	def get_redshifted_spectrum(self, redshift):
		return GaussianSpectrum(self.center_wavelength * (1 + redshift), self.spectral_width * (1 + redshift))

	def approximate(self, num_wavelength, min_wavelength, max_wavelength):
		distr = scipy.stats.norm
		cdf_min = distr.cdf(min_wavelength, loc=self.center_wavelength, scale=self.spectral_width)
		cdf_max = distr.cdf(max_wavelength, loc=self.center_wavelength, scale=self.spectral_width)

		step = (cdf_max - cdf_min) / num_wavelengths
		cdfs = (0.5 + np.arange(num_wavelengths)) * step

		wavelengths = distr.ppf(cdfs, loc=self.center_wavelength, scale=self.spectral_width)
		power = np.ones(len(wavelengths)) * step * self.num_photons_per_second

		wl_grid = CartesianGrid(SeparatedCoords([wavelengths]))
		return TabulatedSpectrum(Field(power, wl_grid))

class LaserSpectrum(AnalyticalSpectrum):
	def __init__(self, laser_wavelength):
		self.laser_wavelength = laser_wavelength
	
	def get_redshifted_spectrum(self, redshift):
		return LaserSpectrum(self.laser_wavelength * (1 + redshift))
	
	def approximate(self, num_wavelength, min_wavelength, max_wavelength):
		wl_grid = CartesianGrid(UnstructuredCoords([[self.laser_wavelength]]), [1])
		return TabulatedSpectrum(Field([self.num_photons_per_second], wl_grid))

class InterpolatableCatalogSpectrum(TabulatedSpectrum):
	def __init__(self, catalog_name, effective_temperature, metalicity, log_g):
		if 'CDBS' in Configuration():
			path = Configuration()['CDBS']:
		elif 'PYSYN_CDBS' in os.environ:
			path = os.environ['PYSYN_CDBS']
		else:
			raise RuntimeError("Couldn't find the Calibration Reference Data System directory. Please configure it.")
		
		cat = path + 0

class PhoenixSpectrum(TabulatedSpectrum):
	def __init__(self, temperature, log_g):
		if 'CDBS' in Configuration():
			path = Configuration()['CDBS']:
		elif 'PYSYN_CDBS' in os.environ:
			path = os.environ['PYSYN_CDBS']
		else:
			raise RuntimeError("Couldn't find the Calibration Reference Data System directory. Please configure it.")
		
		




class Spectrum():
	def __init__(self, wavelength, spectrum, in_air=False):
		self._is_in_air = in_air
		self._wavelength = wavelength.copy()
		self._R = np.mean(self._wavelength) / (np.mean(self._wavelength[1:] - self._wavelength[:-1]))	# The intrinsic sampling
		self._spectrum = spectrum.copy()
	
		self.spline_coef = interpolate.splrep(self._wavelength, self._spectrum)

	def spectrum(self, wavelengths, velocity=0):
		shear = np.sqrt( (1+velocity)/(1-velocity) )
		
		return interpolate.splev(wavelengths*shear, self.spline_coef)

	# Rename the functions prepare and get_spectrum
	def prepare(self, wmin, wmax, velocities):
		mask = (self._wavelength>=wmin)*(self._wavelength<=wmax)
		self.prepared_wavelengths = self._wavelength[mask] 
		self.M = np.array([self.get_template(self.prepared_wavelengths, velocity) for velocity in velocities]).T

	def velocity_spec(self, velocity_profile):
		return self.M.dot(velocity_profile)

	# Maybe make different interface?
	def get_wavelengths(self, wmin, wmax):
		mask = (self._wavelength>=wmin)*(self._wavelength<=wmax)
		return self._wavelength[mask].copy()

	def subset(self, wmin, wmax):
		mask = (self._wavelength>=wmin)*(self._wavelength<=wmax)
		mask_wavelengths = self._wavelength[mask]
		mask_spectrum = self._spectrum[mask]

		return Spectrum(mask_wavelengths, mask_spectrum)

	def degraded(self, wmin, wmax, R, oversamp=2):
		
		# Determine the sampling
		vmax = 3/R
		dv = 1/(2*self._R)
		num_steps = int(2*vmax/dv)

		# Setup the convolution kernel
		velocities = np.linspace(-vmax, vmax, num_steps)
		new_sigma = 1/(2.355*R)
		old_sigma = 1/(2.355*self._R)
		sigma = np.sqrt( new_sigma**2 - old_sigma**2 )

		velocity_profile = np.exp(-0.5 * (velocities/sigma)**2) / np.sqrt(2*np.pi * sigma**2)
		velocity_profile /= np.sum(velocity_profile)

		# Build up the library
		mask = (self._wavelength>=wmin)*(self._wavelength<=wmax)
		mask_wavelengths = self._wavelength[mask]

		max_shear = np.sqrt( (1+vmax)/(1-vmax) )
		new_max = max_shear * wmax
		new_min = wmin/max_shear
		spectrum_builder = self.subset(new_min, new_max)

		#spec_test = spectrum_builder.spectrum(mask_wavelengths, 0)
		
		# Build up the lower resolution spectrum
		start = time.time()
		
		convolved_spectrum = 0
		for ai, velocity in zip(velocity_profile, velocities):
			convolved_spectrum += ai * spectrum_builder.spectrum(mask_wavelengths, velocity)

		end = time.time()
		print("Elapsed {:g}".format(end-start))

		# Bin the spectrum down
		Rsamp = oversamp*R
		dwave = (wmax+wmin)/(2*Rsamp)
		num_waves = int((wmax-wmin)/dwave)
		
		wave_bins = np.linspace(wmin-2*dwave, wmax+dwave, num_waves+3) + dwave/2

		new_spectrum, bin_edges, hist = binned_statistic( mask_wavelengths, convolved_spectrum, statistic='mean', bins = wave_bins )
		new_wave_centers = (wave_bins[2:-1]+wave_bins[1:-2])/2
		
		#print("The new spectrum")
		#plt.plot(new_wave_centers, new_spectrum[1:-1])
		#plt.show()

		# Return a new spectrum class
		return Spectrum(new_wave_centers, new_spectrum[1:-1])

class PhoenixSpectrum(Spectrum):
	'''
		Phoenix Spectra are in Angstrom
	'''
	def __init__(self, spectrum_filename, wavelength_filename):
		spectrum = fits.getdata(spectrum_filename)
		wavelength = fits.getdata(wavelength_filename)
		super().__init__(wavelength, spectrum, False)

class KuruzcSpectrum(Spectrum):
	'''
		Kuruzc Spectra are in Angstrom
	'''
	def __init__(self, filename, continuum_normalized=False):

		# Load data
		data = fits.getdata(filename)
		header = fits.getheader(filename)

		# Construct the wavelength grid
		w0 = header['CRVAL1']
		dw = header['CD1_1']
		wavelength = np.arange(header['NAXIS1']) * dw + w0
		
		if continuum_normalized:
			super().__init__(wavelength, data[0,:], True)
		else:
			super().__init__(wavelength, data[1,:], True)

class BTSettlSpectrum(Spectrum):
	'''
		BTSettl Spectra are in microns
	'''
	def __init__(self, spectrum_filename):
		data = fits.getdata(spectrum_filename)
		spectrum = data.field(1)
		wavelength = data.field(0) * 10000

		super().__init__(wavelength, spectrum)
