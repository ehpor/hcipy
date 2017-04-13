import numpy as np
from .propagator import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field
from ..fourier import FastFourierTransform


# For anti-aliasing the refractive index profile should be sub-sampled
# TODO: add a nicer way to get create 3D electric fields!

class BeamPropagatorMonochromatic(object):
	def __init__(self, input_grid, material, distance, num_steps, num_save=None, verbose=False, wavelength=1):
		self.input_grid = input_grid
		self.fft = FastFourierTransform(input_grid)
		
		self.k = 2*np.pi / wavelength
		self.k_squared = self.fft.output_grid.as_('polar').r**2
		
		self.material = material
		self.distance = distance
		self.num_steps = num_steps
		
		self.verbose = verbose
		self.num_save = num_save
		if not num_save is None:
			self.save_freq = int(self.num_steps/self.num_save)

	def refract(self, wavefront, z, dz):
		# Get refractive index of material at current position
		n = self.material( self.input_grid, z )
		# Seperate mean of profile from variation
		n_dc = np.mean(n)
		n_ac = n - n_dc
		# Create transfer function
		refraction_function = np.exp(1j * self.k * n_ac * dz)

		return Wavefront(wavefront.electric_field * refraction_function, wavefront.wavelength)

	def diffract(self, wavefront, z, dz):
		# Get refractive index of material at current position
		n = self.material( self.input_grid, z )
		# Seperate mean of profile from variation
		n_dc = np.mean(n)
		# Create the transfer function
		k_z = np.sqrt( (n_dc*self.k)**2 - self.k_squared + 0j )         
		self.diffraction_function = np.exp( 1j * k_z * dz)
		# Perform diffraction
		electric_field_out = self.fft.backward(self.fft.forward(wavefront.electric_field) * self.diffraction_function)

		return Wavefront(electric_field_out, wavefront.wavelength)

	def forward(self, wavefront):
		dz = self.distance / self.num_steps
		# Use a second order symplectic integrator for the beam propagation
		# This is a leapfrog algorithm so start the refraction off by half a step
		if not self.num_save is None:
			wavefronts = []

		wavefront_out = self.refract( wavefront, dz/2, dz)
		for i in range(self.num_steps):
			wavefront_temp = self.diffract(wavefront_out, i * dz, dz)
			wavefront_out = self.refract(wavefront_temp, (i+1.5) * dz, dz)
			
			if not self.num_save is None and i % self.save_freq == 0:
				if self.verbose:
					print("step {:d}/{:d}".format(i+1, self.num_steps))
				wavefronts.append( wavefront_out )
		
		if self.num_save is None:
			return wavefront_out
		else:
			return wavefronts
			
	def backward(self, wavefront):
		dz = self.distance/self.num_steps
		# Use a second order symplectic integrator for the beam propagation
		# This is a leapfrog algorithm so start the refraction off by half a step
		wavefront_out = self.refract( wavefront, distance, -dz/2)
		for i in range(num_steps): 
		   wavefront_out = self.refract(self.diffract(wavefront_out, distance - (i + 0.5) * dz, -dz), distance - (i + 1.5) * dz, -dz)
		return wavefront_out

BeamPropagator = make_propagator(BeamPropagatorMonochromatic)