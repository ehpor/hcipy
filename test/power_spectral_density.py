from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def psd(u_grid):
	return (u_grid.as_('polar').r < 100).astype('float') * (u_grid.as_('polar').r > 30).astype('float')

def psd_1d(u_grid):
	return (np.abs(u_grid.x) < 100).astype('float') * (np.abs(u_grid.x) > 30).astype('float')

def make_pupil_grid_1d(N, D=1):
	D = (np.ones(1) * D).astype('float')
	N = (np.ones(1) * N).astype('int')

	delta = D / (N-1)
	zero = -D/2

	return CartesianGrid(RegularCoords(delta, N, zero))

ratios = []
for i in range(100):
	q = 1
	pupil_grid = make_pupil_grid(128, q)
	#pupil_grid = make_pupil_grid_1d(2048, q)
	print(pupil_grid.ndim)

	oversampling = 2
	factory = SpectralNoiseFactoryFFT(psd, pupil_grid, oversampling)
	screen = factory.make_random()()

	power_in_screen = np.sum(np.abs(screen)**2 * screen.grid.weights) / np.sum(screen.grid.weights)
	power_in_psd = np.sum(psd(factory.input_grid) * factory.input_grid.weights / (2*np.pi)**pupil_grid.ndim)

	print('Power in screen:', power_in_screen)
	print('Power in PSD:', power_in_psd)
	print('Ratio:', power_in_screen / power_in_psd)
	ratios.append(power_in_screen / power_in_psd)


plt.hist(ratios, bins=30)
plt.show()
#imshow_field(screen)
#plt.show()