from hcipy import *
from matplotlib import pyplot as plt
import numpy as np

def test_new_pyramid():
	pupil_grid = make_pupil_grid(128)

	num_zernike = 9
	zernike_basis = make_zernike_basis(num_zernike, 1, pupil_grid, 2)

	num_pupil_pixels = 32
	pupil_separation = 1.5
	material = lambda wl: 1.45
	num_pupil_pixels = 32

	aperture = circular_aperture(1)

	pyramid = PyramidWavefrontSensorNew(pupil_grid, aperture, pupil_separation, num_pupil_pixels, 4, 1e-6, material, PerfectDetector)
	imshow_field(pyramid.pyramid.surface)
	plt.colorbar()
	plt.show()

	for i, zern in enumerate(zernike_basis):
		wf = Wavefront(aperture(pupil_grid) * np.exp(1j * zern), 1e-6)
		wf.total_power = 1

		pyramid.integrate(wf)
		shifts = pyramid.read_out()

		plt.figure(1)
		plt.subplot(3,3,i+1)
		imshow_field(pyramid.detector_image)
		plt.colorbar()

		plt.figure(2)
		plt.subplot(3,3,i+1)
		imshow_field(shifts[:,0])
		plt.colorbar()

		plt.figure(3)
		plt.subplot(3,3,i+1)
		imshow_field(shifts[:,1])
		plt.colorbar()
	
	plt.show()

if __name__ == "__main__":
	#test_new_pyramid()
	#exit()

	pupil_plane = make_pupil_grid(128,1)

	num_zernike = 9
	zernike_basis = make_zernike_basis(num_zernike, 1, pupil_plane, 2)

	pupil_separation = 1.5
	num_pupil_pixels = 32
	pyramid = PyramidWavefrontSensor(pupil_plane, pupil_separation, num_pupil_pixels, PerfectDetector, over_sampling=4, wavelength=1.0E-6)

	aperture = circular_aperture(1)(pupil_plane)
	for i in range(num_zernike):
		
		wf = Wavefront(aperture * np.exp(1j * zernike_basis[i]), wavelength=1.0E-6 )
		wf.total_power = 1
		
		pyramid.integrate(wf)
		pupil_intensity = pyramid.detector.read_out() 
		reduced_pupils = pyramid.reduced_pupils(pupil_intensity)
		measurement = pyramid.measurement(pupil_intensity)

		plt.figure(1)
		plt.subplot(3,3,i+1)
		imshow_field(pupil_intensity)
		plt.colorbar()
		
		plt.figure(2)
		plt.subplot(3,3,i+1)
		plt.imshow(reduced_pupils[0])
		plt.colorbar()

		plt.figure(3)
		plt.subplot(3,3,i+1)
		plt.imshow(reduced_pupils[1])
		plt.colorbar()

	plt.show()