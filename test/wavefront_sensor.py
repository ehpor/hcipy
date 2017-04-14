from hcipy import *
from matplotlib import pyplot
import numpy as np

if __name__ == "__main__":
	pupil_plane = make_pupil_grid(128,1)
	output_grid = make_pupil_grid(64,4.0)

	Nz = 5
	zernike_basis = make_zernike_basis(Nz, 1, pupil_plane, 2)
	if False:
		for i,m in enumerate(zernike_basis):
			pyplot.subplot(3,3,i+1)
			imshow_field(zernike_basis[i], cmap='RdBu')
			pyplot.axis('off')
		pyplot.show()
	
	pyramid = PyramidWavefrontSensor(pupil_plane,output_grid,CCD(output_grid),wavelength=1.0E-6)

	aperture = circular_aperture(1)(pupil_plane)
	for i in range( Nz ):
		
		wf = Wavefront( aperture * numpy.exp(1j * zernike_basis[i]), wavelength = 1.0E-6 )
		wf.total_power = 1
		pyramid.integrate(wf)
		It = pyramid.detector.read_out() 
		pyplot.figure(1)
		pyplot.subplot(3,3,i+1)
		imshow_field( It )
		pyplot.colorbar()
		
		pyplot.figure(2)
		pyplot.subplot(3,3,i+1)
		pyplot.imshow( pyramid.measurement(It)[0] )
		pyplot.colorbar()

		pyplot.figure(3)
		pyplot.subplot(3,3,i+1)
		pyplot.imshow( pyramid.measurement(It)[1] )
		pyplot.colorbar()

	pyplot.show()