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
F_mla = 0.01
N_mla = 20

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
amp = 1
Influence = shack_hartmann_calibrator(wf, shwfs, shwfse, det, zernike_freeform, amp)

## Extract the centers of the lenslets
act_levels = np.zeros(num_modes)
zernike_freeform.actuators = act_levels
lenslet_centers = shwfse.estimate([shwfs.forward(zernike_freeform.forward(wf)).intensity])

## Calculate the DM response matrix
Response = inverse_truncated(Influence, 1e-3)

act_levels = np.zeros(num_modes)
zernike_freeform.actuators = act_levels

## Start loop of timesteps here
times = np.linspace(0,0.7,200)
mw = GifWriter('shack_hartmann.gif', 10)
for t in times:
        
    ## Generate phase screen to simulate atmospheric turbulence
    atmospheric_model.t = t
    wfdistort = atmospheric_model(wf)
    det.integrate(prop(wfdistort), 1, 1)
    distort_img = det.read_out()
    
    ## Propagate wavefront through the wavefront sensor
    wfao = shwfs.forward(wfdistort)
    det.integrate(wfao, 1, 1)
    ao_img = det.read_out()

    ## Estimate the shifts from the microlens image
    imshift = shwfse.estimate([ao_img]) - lenslet_centers

    ## Find the phase of the wavefront by reconstruction
    act_levels = np.squeeze(np.dot(Response, imshift))

    ## Add modes to DM to correct the wavefront
    zernike_freeform.actuators -= act_levels

    ## Propagate the wavefront onto the DM
    wfcorrect = zernike_freeform.forward(wfdistort)
    
    det.integrate(prop(wfcorrect), 1, 1)
    sci_img = det.read_out()
    
    plt.clf()
    ## Record the following:
    ## - The atmosphere phase screen
    plt.subplot(2,2,1)
    imshow_field(wfdistort.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
    ## - The wavefront after ao correction
    plt.subplot(2,2,2)
    imshow_field(wfcorrect.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
    ## - The psf of the distorted wavefront
    plt.subplot(2,2,3)
    imshow_field(np.log10(distort_img / distort_img.max()), vmin=-5)
    ## - The science image
    plt.subplot(2,2,4)
    imshow_field(np.log10(sci_img / sci_img.max()), vmin=-5)
    
    mw.add_frame()
mw.close()