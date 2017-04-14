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

class PyramidWavefrontSensor(WavefrontSensor):
	def __init__(self, input_grid, output_grid, detector, wavelength):
		self.detector = detector
		self.output_grid = output_grid
		self.input_grid = input_grid

		self.optical_system = self.make_optical_system(self.input_grid, self.output_grid,wavelength)

	def make_optical_system(self, input_grid, output_grid,wavelength):
		fourier_grid = make_focal_grid(input_grid,q=4,wavelength=wavelength)

		fraunhofer_1 = FraunhoferPropagator(input_grid,fourier_grid,wavelength_0=wavelength)
		a = 1.0 * 2.0 * np.pi/wavelength
		b = 1.0 * 2.0 * np.pi/wavelength
		
		T = quadrant_filter( np.array([1.0,1.0]) , np.array([0.5,0.5]) )(fourier_grid) * np.exp(1j* (a * fourier_grid.x + b * fourier_grid.y))
		T += quadrant_filter( np.array([1.0,1.0]) , np.array([0.5,-0.5]) )(fourier_grid) * np.exp(1j* (a * fourier_grid.x - b * fourier_grid.y))
		T += quadrant_filter( np.array([1.0,1.0]) , np.array([-0.5,0.5]) )(fourier_grid) * np.exp(1j* (-a * fourier_grid.x + b * fourier_grid.y))
		T += quadrant_filter( np.array([1.0,1.0]) , np.array([-0.5,-0.5]) )(fourier_grid) * np.exp(-1j* (a * fourier_grid.x + b * fourier_grid.y))

		pyramid_prism = Apodizer(T.flatten())
		fraunhofer_2 = FraunhoferPropagator(fourier_grid, output_grid, wavelength_0=wavelength)

		A = Field( np.angle(T.flatten()), fourier_grid )
		#imshow_field( A )
		#pyplot.colorbar()
		#pyplot.show()

		temp = OpticalSystem()
		temp.add_element(fraunhofer_1)
		temp.add_element(pyramid_prism)
		temp.add_element(fraunhofer_2)

		return temp

	def measurement(self, pupil_intensity):
		print( pupil_intensity.grid.shape )
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