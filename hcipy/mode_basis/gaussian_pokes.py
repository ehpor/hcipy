
from ..field import Field
from .mode_basis import ModeBasis
import numpy as np

def make_gaussian_pokes(grid, mu, sigma):
	sigma = np.ones(mu.size) * sigma
	return ModeBasis([Field(np.exp(-0.5*grid.shifted(p).as_('polar').r**2/s**2), grid) for p, s in zip(mu.points, sigma)])

'''
from hcipy import *
import matplotlib.pyplot as plt

pupil_grid = make_pupil_grid(256, 5)
mu = make_hexagonal_grid(0.5, 5)#make_pupil_grid(11,4).subset(circular_aperture(4))
sigma = 2./11
g = make_gaussian_pokes(pupil_grid, mu, sigma)

plt.plot(mu.x, mu.y, 'o', c=colors.blue)
plt.figure()

for i, m in enumerate(g):
	plt.subplot(11,11,i+1)
	imshow_field(m)
	plt.colorbar()
plt.show()
'''