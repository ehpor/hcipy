from .wavefront_sensor import WavefrontSensor
from ..optics import Apodizer
from ..propagation import FraunhoferPropagator
from ..aperture import rectangular_aperture
from ..field.util import *
from ..field import *
from ..plotting import *
from ..optics import OpticalSystem

from matplotlib import pyplot
import numpy as np

def quadrant_filter(size, center=None):
	dim = size * np.ones(2)
	if center is None:
		shift = np.zeros(2)
	else:
		shift = center * np.ones(2)
	
	def func(grid):
		x, y = grid.as_('cartesian').coords
		f = (np.abs(x/x.max()-shift[0]) <= (dim[0]/2)) * (np.abs(y/y.max()-shift[1]) <= (dim[1]/2))	
		return f.astype('float')
	
	return func

# TODO: fix quadrant code!
# TODO: change input variables.
# change output_grid to pupil_diameter and pupil_separation
class PyramidWavefrontSensor(WavefrontSensor):
	def __init__(self, input_grid, output_grid, detector, wavelength):
		
		self.output_grid = output_grid
		self.input_grid = input_grid
		self.detector = detector(output_grid)

		self.optical_system = self.make_optical_system(self.input_grid, self.output_grid,wavelength)

	def make_optical_system(self, input_grid, output_grid,wavelength):
		fourier_grid = make_focal_grid(input_grid,q=16,wavelength=wavelength)

		fraunhofer_1 = FraunhoferPropagator(input_grid,fourier_grid,wavelength_0=wavelength)
		a = 0.5 * 2.0 * np.pi/wavelength
		b = 0.5 * 2.0 * np.pi/wavelength
		
		T = (fourier_grid.x>0) * (fourier_grid.y>0) * np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y))
		T += (fourier_grid.x>0) * (fourier_grid.y<0) * np.exp(1j* (a * fourier_grid.x - b * fourier_grid.y))
		T += (fourier_grid.x<0) * (fourier_grid.y>0) * np.exp(1j* (-a * fourier_grid.x + b * fourier_grid.y))
		T += (fourier_grid.x<0) * (fourier_grid.y<0) * np.exp(-1j* (a * fourier_grid.x + b * fourier_grid.y))

		horizontal_edge_mask = abs(fourier_grid.x)<1.0E-13
		vertical_edge_mask = abs(fourier_grid.y)<1.0E-13
		T[horizontal_edge_mask] += 0.5 * (np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y)) + np.exp(1j* (a * fourier_grid.x - b * fourier_grid.y)))[horizontal_edge_mask]
		T[vertical_edge_mask] += 0.5 * (np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y)) + np.exp(1j* (-a * fourier_grid.x + b * fourier_grid.y)))[vertical_edge_mask]
		T = T/np.abs(T)

		pyramid_prism = Apodizer(T)
		fraunhofer_2 = FraunhoferPropagator(fourier_grid, output_grid, wavelength_0=wavelength)

		if False:
			A = Field( np.angle(T.flatten()), fourier_grid )
			B = Field( np.abs(T.flatten()), fourier_grid)
			pyplot.subplot(1,2,1)
			imshow_field( A )
			pyplot.colorbar()
			pyplot.subplot(1,2,2)
			imshow_field( B )
			pyplot.colorbar()
			pyplot.show()

		temp = OpticalSystem((fraunhofer_1,pyramid_prism,fraunhofer_2))

		return temp

	def measurement(self, pupil_intensity):
		B = np.reshape( pupil_intensity, pupil_intensity.grid.shape )
		shape = B.shape

		Nx = shape[0]//2
		Ny = shape[0]//2
		Ia = B[0:Nx,0:Ny]
		Ib = B[Nx:(2*Nx),0:Ny]
		Ic = B[Nx:(2*Nx),Ny:(2*Ny)]
		Id = B[0:Nx,Ny:(2*Ny)]

		I1 = (Ia+Ib-Ic-Id)/(Ia+Ib+Ic+Id)
		I2 = (Ia-Ib-Ic+Id)/(Ia+Ib+Ic+Id)
		return I1, I2