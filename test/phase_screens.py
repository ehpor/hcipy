from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time

from hcipy.atmosphere import *

pupil_grid = make_pupil_grid(256, 2)
focal_grid = make_focal_grid(pupil_grid, 8, 16)

r0 = 0.3
wavelength = 500e-9
L0 = 10
velocity = 3
height = 0
stencil_length = 2
oversampling = 128

layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, height, stencil_length)
#layer = FiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, height, oversampling)

start = time.time()
N = 1000
for i in range(N):
	layer._extrude('left')
end = time.time()

print('columns per sec:', N / (end - start))
print('phase screens per sec:', N / (end - start) / pupil_grid.shape[0])

layers = []
layers.append(InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length))
layers.append(InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 500, stencil_length))

atmosphere = MultiLayerAtmosphere(layers, True)
atmosphere.Cn_squared = Cn_squared_from_fried_parameter(1, wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid.scaled(wavelength))

aperture = circular_aperture(1)(pupil_grid)
wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)

for t in np.linspace(0, 100, 5001):
	atmosphere.evolve_until(t)

	wf2 = atmosphere.forward(wf)
	wf2.electric_field *= aperture
	img = Field(prop(wf2).intensity, focal_grid)
	
	plt.clf()
	plt.subplot(1,2,1)
	imshow_field(wf2.intensity, cmap='gray')
	plt.subplot(1,2,2)
	imshow_field(img)
	plt.draw()
	plt.pause(0.00001)