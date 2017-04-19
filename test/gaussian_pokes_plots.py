# -*- coding: utf-8 -*-

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