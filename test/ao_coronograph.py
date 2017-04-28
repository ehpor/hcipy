# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:36:19 2017

@author: radhakrishnan

@description: Optical setup consisting of a simple Shack-Hartmann wavefront
sensor with deformable mirror, and an APP style coronograph. Atmospheric
turbulence is simulated and its effects on the dark hole are recorded.
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
F_mla = 0.03
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

### Calibrate the DM influence matrix
#num_modes = 60
#dm_modes = make_zernike_basis(num_modes, 1, pupil_grid, 1, False)
#zernike_freeform = DeformableMirror(dm_modes)
#act_levels = np.zeros(num_modes)
#amp = 0.1
#Influence = []
#
#zernike_freeform.actuators = act_levels
#lenslet_centers = shwfse.estimate([shwfs.forward(zernike_freeform.forward(wf)).intensity])
#
#shmw = GifWriter('sh_images.gif', 10)
#for act in np.arange(num_modes):
#    ## Set each actuator to +1 in sequence
#    act_levels = np.zeros(num_modes)
#    act_levels[act] = amp
#    zernike_freeform.actuators = act_levels
#    
#    ## Propagate the wf on the deformable mirror
#    wfdm = zernike_freeform.forward(wf)
#    
#    ## Propagate the wavefront through the Shack-Hartmann optics
#    wfmla = shwfs.forward(wfdm)
#    wfmla_img = wfmla.intensity
#    
#    Splus = shwfse.estimate([wfmla_img])
#    
#    plt.clf()
#    ## Record the following:
#    ## - The wavefront imposed by the DM
#    plt.subplot(1,2,1)
#    imshow_field(wfdm.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
#    ## - The microlens array image plane
#    plt.subplot(1,2,2)
#    imshow_field(np.log10(wfmla_img / wfmla_img.max()), vmin=-5)
#    
#    shmw.add_frame()
#    
#    # Set each actuator to -1 in sequence
#    act_levels[act] = -amp
#    zernike_freeform.actuators = act_levels
#    
#    ## Propagate the wf on the deformable mirror
#    wfdm = zernike_freeform.forward(wf)
#    
#    ## Propagate the wavefront through the Shack-Hartmann optics
#    wfmla = shwfs.forward(wfdm)
#    wfmla_img = wfmla.intensity
#    
#    Sminus = shwfse.estimate([wfmla_img])
#    
#    plt.clf()
#    ## Record the following:
#    ## - The wavefront imposed by the DM
#    plt.subplot(1,2,1)
#    imshow_field(wfdm.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
#    ## - The microlens array image plane
#    plt.subplot(1,2,2)
#    imshow_field(np.log10(wfmla_img / wfmla_img.max()), vmin=-5)
#    
#    shmw.add_frame()
#    
#    ## Calculate the shift for this mode and append it to the influence matrix
#    shift = (Splus - Sminus) / (2 * amp)
#    Influence.append(shift)
#    
#shmw.close()

## Calibrate the DM influence matrix
amp = 0.1
Influence = shack_hartmann_calibrator(wf, shwfs, shwfse, det, zernike_freeform, amp)
lenslet_centers = shwfse.estimate([shwfs.forward(zernike_freeform.forward(wf)).intensity])

Inf_modes = ModeBasis(Influence)
Response = inverse_truncated(Inf_modes.transformation_matrix)


act_levels = np.zeros(num_modes)
zernike_freeform.actuators = act_levels

## Start loop of timesteps here
times = np.linspace(0,0.7,200)
mw = GifWriter('ao_coronograph.gif', 10)
for t in times:
        
    ## Generate phase screen to simulate atmospheric turbulence
    atmospheric_model.t = t
    wfdistort = atmospheric_model(wf)
    #wfdistort = wf.copy()
    
    ## Propagate wavefront through the wavefront sensor
    wfao = shwfs.forward(wfdistort)
    det.integrate(wfao, 1, 1)
    ao_img = det.read_out()
    
    ## Estimate the shifts from the microlens image
    imshift = shwfse.estimate([ao_img]) - lenslet_centers
    
    ## Find the phase of the wavefront by reconstruction
    act_levels = np.squeeze(np.dot(Response, imshift))

    ## Add modes to DM to correct the wavefront
    zernike_freeform.actuators = -act_levels
    
    ## Propagate the wavefront onto the DM
    wfcorrect = zernike_freeform.forward(wfdistort)

    ## Propagate wavefront through coronograph
    app_amp = pf.getdata('./coronographs/Square_20_80_20_25_0_2_amp_resampled_256.fits').ravel()
    app_phase = pf.getdata('./coronographs/Square_20_80_20_25_0_2_phase_resampled_256.fits').ravel()
    app_psf = pf.getdata('./coronographs/Square_20_80_20_25_0_2_psf_resampled_256.fits').ravel()
    app = app_amp * np.exp(1j * app_phase)
    
    wfc = wfcorrect.copy()
    wfc.electric_field *= app
    sci_img = prop(wfc).intensity
    
    zernike_decom = dm_modes.transformation_matrix.T.dot(wfdistort.phase)
    
    plt.clf()
    ## Record the following:
    ## - The atmosphere phase screen
    plt.subplot(2,2,1)
    imshow_field(wfdistort.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
    ## - The wavefront after ao correction
    plt.subplot(2,2,2)
    imshow_field(zernike_freeform.surface * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
    
    ## - The microlens array image
    plt.subplot(2,2,3)
    imshow_field(np.log10(ao_img / ao_img.max()), vmin=-5)
    ## - The science image after coronograph
    plt.subplot(2,2,4)
    imshow_field(np.log10(sci_img / sci_img.max()), vmin=-5)
    
    mw.add_frame()
mw.close()
