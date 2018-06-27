# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:36:19 2017

@author: radhakrishnan

@description: Optical setup consisting of a simple Shack-Hartmann wavefront
sensor with deformable mirror. Atmospheric turbulence is simulated.
"""
## General imports
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

## Create aperture and pupil grid
wavelength = 1e-6
N = 256
D = 0.01
pupil_grid = make_pupil_grid(N, D)
focal_grid = make_focal_grid(pupil_grid, 8, 20, wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
aperture = circular_aperture(D)
#aperture = rectangular_aperture(1,1)

## Create the microlens array
F_mla = 100
N_mla = 20

shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid, F_mla, N_mla, D)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

## Create the wavefront at the entrance pupil
wf = Wavefront(aperture(pupil_grid), wavelength)

## Create the deformable mirror
num_modes = 60
dm_modes = make_zernike_basis(num_modes, D, pupil_grid, 2, False)
zernike_freeform = DeformableMirror(dm_modes)

plt.ion()
for i in range(num_modes):
	## Extract the centers of the lenslets
	act_levels = np.zeros(num_modes)
	act_levels[i] = 1e-7
	zernike_freeform.actuators = act_levels
	dm_wf = zernike_freeform(wf)
	sh_wf = shwfs(dm_wf)
	sh_img = sh_wf.intensity
	lenslet_centers = shwfse.estimate([sh_img])

	plt.clf()
	imshow_field(sh_img)
	plt.plot(lenslet_centers[0,:], lenslet_centers[1,:], 'r,')
	plt.colorbar()
	plt.draw()
	plt.pause(0.00001)
plt.ioff()