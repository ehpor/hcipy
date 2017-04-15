from hcipy import *
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
	pupil_plane = make_pupil_grid(128,1)

	num_zernike = 5
	zernike_basis = make_zernike_basis(num_zernike, 1, pupil_plane, 2)

	pupil_separation = 1
	num_pupil_pixels = 16
	pyramid = PyramidWavefrontSensor(pupil_plane, pupil_separation, num_pupil_pixels, CCD, over_sampling=4, wavelength=1.0E-6)

	aperture = circular_aperture(1)(pupil_plane)
	for i in range(num_zernike):
		
		wf = Wavefront(aperture * np.exp(1j * zernike_basis[i]), wavelength=1.0E-6 )
		wf.total_power = 1
		pyramid.integrate(wf)
		It = pyramid.detector.read_out() 
		plt.figure(1)
		plt.subplot(3,3,i+1)
		imshow_field(It)
		plt.colorbar()
		
		plt.figure(2)
		plt.subplot(3,3,i+1)
		plt.imshow(pyramid.measurement(It)[0])
		plt.colorbar()

		plt.figure(3)
		plt.subplot(3,3,i+1)
		plt.imshow(pyramid.measurement(It)[1])
		plt.colorbar()

	plt.show()