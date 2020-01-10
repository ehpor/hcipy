from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time

from hcipy.atmosphere import *

pupil_grid = make_pupil_grid(256, 1)
focal_grid = make_focal_grid(pupil_grid, 8, 16)

r0 = 0.3
wavelength = 500e-9
L0 = 10
velocity = 3
height = 0
stencil_length = 2
oversampling = 128

mode_basis = make_zernike_basis(500, 1, pupil_grid, 1)

layers = []
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
layer2 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
layers.append(layer2)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
layer3 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
layers.append(layer3)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
layer4 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
layers.append(layer4)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
layer5 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
layers.append(layer5)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
layer6 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
layers.append(layer6)

atmosphere = MultiLayerAtmosphere(layers, False)
atmosphere.Cn_squared = Cn_squared_from_fried_parameter(1/40, wavelength)
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
	imshow_field(wf2.phase, cmap='RdBu')
	plt.subplot(1,2,2)
	imshow_field(np.log10(img / img.max()), vmin=-6)
	plt.draw()
	plt.pause(0.00001)