from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

wavelength = 1e-6 # m
D_tel = 100 * wavelength
N_over = 4
F_num = 3
distance = D_tel * F_num
k = 2*np.pi / wavelength

N_samp = 1024
pupil_grid = make_pupil_grid(N_samp, D_tel*N_over)
aperture = circular_aperture(D_tel)
focal_grid = make_focal_grid(pupil_grid, 1, 500)

fresnel = FresnelPropagator(pupil_grid, distance)
ASP = AngularSpectrumPropagator(pupil_grid, distance)
props = [fresnel, ASP]

aper = aperture(pupil_grid) * np.exp(-1j * k * pupil_grid.as_('polar').r**2 / (2 * distance))
tilts = ((0, 0), (10, 10), (20, 20))
Ntilts = len(tilts)

for j, tilt in enumerate(tilts):

	phase_tilt = 2*np.pi * (pupil_grid.x / D_tel * tilt[0] + pupil_grid.y / D_tel * tilt[1])
	aper *= np.exp(1j * phase_tilt)

	wf = Wavefront(aper, wavelength)
	wf.total_power = 1

	for i, prop in enumerate(props):
		img = prop(wf)

		plt.subplot(Ntilts, 2, 2*j + i + 1)
		imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-5)
plt.show()