
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


## Create the wavefront at the entrance pupil
wf = Wavefront(aperture(pupil_grid), wavelength)
wf.total_power = 1

prop = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=wavelength)

## Create the microlens array for wavefront sensor
F_mla = 30
N_mla = 11
d=D/N_mla

shwfs = SquareShackHartmannWavefrontSensorOptics(shwfs_mask, F_mla, N_mla, D)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

## Create the deformable mirror
num_modes = 40
modes = make_zernike_basis(num_modes, D, pupil_grid, 2, False)
zernike_freeform = DeformableMirror(modes)

##Create our filter to do the reconstruction
n = 40
modes_filter = make_zernike_basis(n, D, pupil_grid, 2, False)
my_filter=ModalReconstructor(modes_filter)
calibrate_modal_reconstructor(my_filter,n,wf,shwfs,shwfse)

zernike_freeform.actuators=np.zeros(num_modes)
zernike_freeform.actuators[10]=wavelength*0.075
wf1=zernike_freeform.forward(wf)
#print(np.sqrt(np.mean(np.square(wf1.phase))))
#measure the wavefront and filter it to see what we get
img = shwfs(wf1).intensity
slopes = shwfse.estimate([img]).ravel()

measured_zernike=my_filter.estimate(slopes,0)

plt.figure()
plt.plot(measured_zernike,'*',label='Measured')
plt.plot(zernike_freeform.actuators/wavelength,label='Real')
plt.legend()
plt.show()
