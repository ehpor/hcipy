from hcipy import *
from matplotlib import pyplot as plt
import numpy as np

def test_fourier_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	num_zernike = 9
	zernike_basis = make_zernike_basis(num_zernike, 1, pupil_grid, 2)

	num_pupil_pixels = 32
	pupil_separation = 2
	material = lambda wl: 1.45
	num_pupil_pixels = 32
	wavelength_0 = 1e-6

	aperture = circular_aperture(1)

	wfs_detector = PerfectDetector()

	# Create wavefront sensors
	pyramid_wfs = PyramidWavefrontSensorOptics(pupil_grid, pupil_grid.x.ptp(), pupil_separation, num_pupil_pixels, 8, wavelength_0, material)
	zernike_wfs = ZernikeWavefrontSensorOptics(pupil_grid, 1, num_pupil_pixels, q=16, wavelength_0=wavelength_0, diameter=1, num_airy=10)
	rooftop_wfs = RooftopWavefrontSensorOptics(pupil_grid, 1, pupil_separation, num_pupil_pixels, 8, wavelength_0, material)
	gOD_wfs = PolgODWavefrontSensorOptics(1/3, pupil_grid, 1, pupil_separation, num_pupil_pixels, 8, wavelength_0, material)
	wavefront_sensors = (pyramid_wfs, rooftop_wfs, gOD_wfs, zernike_wfs)
		
	# Create reference for zernike wfs
	wf = Wavefront(aperture(pupil_grid), 1.5e-6)
	wf.total_power = 1
	
	wf_ref = zernike_wfs.forward(wf)
	wfs_detector.integrate( wf_ref, 1)
	zernike_reference = wfs_detector.read_out()
	references = (0, 0, 0, zernike_reference)
	
	# Create estimators for the wavefront sensors 
	pyramid_estimator = PyramidWavefrontSensorEstimator(aperture, pyramid_wfs.output_grid)
	zernike_estimator = ZernikeWavefrontSensorEstimator(aperture, zernike_wfs.output_grid, zernike_reference)
	od_estimator = OpticalDifferentiationWavefrontSensorEstimator(aperture, rooftop_wfs.output_grid)
	wavefront_estimators = (pyramid_estimator, od_estimator, od_estimator, zernike_estimator)

	for k, wfs in enumerate(wavefront_sensors):
		for i, zern in enumerate(zernike_basis):
			wf = Wavefront(aperture(pupil_grid) * np.exp(1j * 0.1 * zern), 1.5e-6)
			wf.total_power = 1

			wf_out = wfs.forward(wf)
			print( wf_out.total_power )
			wfs_detector.integrate(wf_out,1)
			wfs_image = wfs_detector.read_out()
			
			plt.figure(k+1)
			plt.subplot(3,3,i+1)
			imshow_field(wfs_image - references[k])
			plt.colorbar()

			test = wavefront_estimators[k].estimate(wfs_image)
			plt.figure(k+10)
			plt.subplot(3,3,i+1)
			plt.plot( test )

	plt.show()

if __name__ == "__main__":
	test_fourier_wavefront_sensor()