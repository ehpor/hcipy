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

def test_old_pyramid():
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

def test_fourier_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	num_zernike = 9
	zernike_basis = make_zernike_basis(num_zernike, 1, pupil_grid, 2)

	num_pupil_pixels = 32
	pupil_separation = 2.0
	material = lambda wl: 1.45
	num_pupil_pixels = 32
	wavelength_0 = 1e-6

	aperture = circular_aperture(1)

	pyramid_wfs = pyramid_wavefront_sensor(pupil_grid, pupil_grid.x.ptp(), pupil_separation, num_pupil_pixels, 4, wavelength_0, material)
	zernike_wfs = zernike_wavefront_sensor(pupil_grid, 1, num_pupil_pixels, q=16, wavelength_0=wavelength_0, diameter=1, num_airy=10)
	rooftop_wfs = rooftop_wavefront_sensor(pupil_grid, 1, pupil_separation, num_pupil_pixels, 4, wavelength_0, material)
	gOD_wfs = gOD_wavefront_sensor(1/3, pupil_grid, 1, pupil_separation, num_pupil_pixels, 4, wavelength_0, material)
	wavefront_sensors = (pyramid_wfs, zernike_wfs, rooftop_wfs, gOD_wfs)

	wfs_detector = PerfectDetector()
	
	# Create reference
	wf = Wavefront(aperture(pupil_grid), 1.5e-6)
	wf.total_power = 1
	
	wf_ref = zernike_wfs.forward(wf)
	wfs_detector.integrate( wf_ref, 1)
	zernike_reference = wfs_detector.read_out()

	references = (0, zernike_reference, 0, 0)
	
	for k, wfs in enumerate(wavefront_sensors):
		for i, zern in enumerate(zernike_basis):
			wf = Wavefront(aperture(pupil_grid) * np.exp(1j * zern), 1.5e-6)
			wf.total_power = 1

			wf_out = wfs.forward(wf)
			wfs_detector.integrate(wf_out,1)
			wfs_image = wfs_detector.read_out()

			plt.figure(k+1)
			plt.subplot(3,3,i+1)
			imshow_field(wfs_image - references[k])
			plt.colorbar()

	plt.show()

if __name__ == "__main__":
	test_fourier_wavefront_sensor()