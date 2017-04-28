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
import pyfits as pf

## Create aperture and pupil grid
r_0 = 0.1
N = 256
pupil_grid = make_pupil_grid(N, 1)
focal_grid = make_focal_grid(pupil_grid, 8, 20)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
aperture = circular_aperture(1)
#aperture = rectangular_aperture(1,1)

## Create the microlens array
F_mla = 20
N_mla = 8

shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid, F_mla, N_mla)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_index)

## Create the detector
det = PerfectDetector()

## Create the wavefront at the entrance pupil
wf = Wavefront(aperture(pupil_grid))

## Create the deformable mirror
num_modes = 60
dm_modes = make_zernike_basis(num_modes, 1, pupil_grid, 2, False)
zernike_freeform = DeformableMirror(dm_modes)

## Create atmospheric turbulence model
spectral_noise_factory = SpectralNoiseFactoryFFT(kolmogorov_psd, pupil_grid, 8)
turbulence_layers = make_standard_multilayer_atmosphere(r_0, wavelength=1)
atmospheric_model = AtmosphericModel(spectral_noise_factory, turbulence_layers)

## Calibrate the DM influence matrix
amp = 0.1
Influence = shack_hartmann_calibrator(wf, shwfs, shwfse, det, zernike_freeform, amp)

## Extract the centers of the lenslets
act_levels = np.zeros(num_modes)
zernike_freeform.actuators = act_levels
lenslet_centers = shwfse.estimate([shwfs.forward(zernike_freeform.forward(wf)).intensity])
