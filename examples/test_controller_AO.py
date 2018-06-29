
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

## Create aperture and pupil grid
wavelength =1e-6
N = 256

D=1000*wavelength
r0=D/10
pupil_grid = make_pupil_grid(N, 1.4*D)
shwfs_mask = make_pupil_grid(N, D) 

focal_grid = make_focal_grid(pupil_grid, q=4, num_airy=20, wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
aperture = circular_aperture(D)

## Create the microlens array
F_mla = 30
N_mla = 11
d=D/N_mla

shwfs = SquareShackHartmannWavefrontSensorOptics(shwfs_mask, F_mla, N_mla, D)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

## Create the wavefront at the entrance pupil
wf = Wavefront(aperture(pupil_grid), wavelength)
wf.total_power = 1

prop = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=wavelength)

## Create the deformable mirror
num_modes = 30
modes = make_zernike_basis(num_modes, D, pupil_grid, 2, False)
zernike_freeform = DeformableMirror(modes)

#create second deformable mirror to inject whatever shape you want 
phase_modes=make_zernike_basis(num_modes, D, pupil_grid, 2, False)
zernike_offsetter = DeformableMirror(phase_modes)

#want to reconstruct the phase and then go to actuator space - need filter with controller
#first make our filter which acts like a reconstructor
my_filter = ModalReconstructor(modes)
calibrate_modal_reconstructor(my_filter,num_modes,wf,shwfs,shwfse)

#set up controller now that takes in the filtered measurements
int_controller = IntegratorController(0.2, interaction_matrix=None,leakage=0.01)
make_interaction_matrix(int_controller,zernike_freeform,wf,shwfs,shwfse,amplitude=0.005,f=my_filter)

#lets get the proper lenslet measurements we want
img = shwfs(wf).intensity
ref = shwfse.estimate([img]).ravel()
num_measurements = ref.shape[0]

plt.ion()
input_wf=0

# Ref psf
psf = prop(wf)
Inorm = psf.intensity.max()

zernike_freeform.actuators = np.zeros(num_modes)
voltages = zernike_freeform.actuators * wavelength

zernike_offsetter.actuators[5] = 0.5
zernike_offsetter.actuators[7] = 0.4
zernike_offsetter.actuators[3] = 0.4
wf.electric_field = aperture(pupil_grid) * np.exp(-1j*zernike_offsetter.surface)
psf = prop(wf)
Inorm = psf.intensity.max()

plt.figure()
imshow_field(wf.phase)
dm_wf = zernike_freeform(wf)
wf_foc = prop.forward(dm_wf)
plt.subplot(2,2,1)
imshow_field(dm_wf.phase) #this is the residual phase -want this to go to zero
plt.colorbar()   
plt.subplot(2,2,2)
imshow_field(wf.phase) #this is the input phase - currently static at the moment
plt.colorbar()
plt.subplot(2,2,3)
imshow_field(img)
plt.subplot(2,2,4)
imshow_field(np.log10(psf.intensity/Inorm +1E-15))
plt.colorbar()

for ii in range(200):
     dm_wf = zernike_freeform.forward(wf)

     sh_wf = shwfs.forward(dm_wf) 
     
     sh_img = sh_wf.intensity
     meas_vec = (shwfse.estimate([sh_img]))
     phase=my_filter.estimate(meas_vec.ravel()-ref,0)
      
     int_controller.submit_wavefront(0, phase, 0, 1)
     zernike_freeform.actuators=int_controller.actuators*wavelength


     #correct using the dm in close loop
     if ii % 1 == 0:
           wf_focus = prop.forward(dm_wf)
           PSF = wf_focus.intensity / Inorm
           plt.clf()
           plt.subplot(2,2,1)
           imshow_field(dm_wf.phase) #this is the residual phase -want this to go to zero
           plt.colorbar()
    
           plt.subplot(2,2,2)
           imshow_field(wf.phase) #this is the input phase - currently static at the moment
           plt.colorbar()
           plt.subplot(2,2,3)
           imshow_field(sh_img)
           plt.subplot(2,2,4)
           imshow_field( np.log10(PSF +1E-15))
           plt.colorbar()
    
           plt.draw()
           plt.pause(0.001)

     



