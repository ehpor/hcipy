from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

wavelength = 1.0E-6
Dtel = 100 * wavelength
Nover = 4
Fnum = 3
distance = Dtel * Fnum

Nsamp = 1024
pupil_grid = make_pupil_grid(Nsamp,D=Dtel*Nover)
aperture = circular_aperture(Dtel)
focal_grid = make_focal_grid(pupil_grid, 1, 0.1)

fresnel = FresnelPropagator(pupil_grid, distance)
ASP = AngularSpectrumPropagator(pupil_grid, distance)
props = [ fresnel, ASP ]

aper = aperture(pupil_grid) * np.exp(-1j * (pupil_grid.as_('polar').r**2)/(2*distance) * 2.0 *np.pi/wavelength )
tilts = ((0.0,0.0),(10.0,10.0),(20.0,20.0))
Ntilts = len(tilts)

for j,tilt in enumerate( tilts ):

	phase_tilt = 2.0 * np.pi * pupil_grid.x/Dtel * tilt[0] + 2.0 * np.pi * pupil_grid.y/Dtel * tilt[1]
	aper *= np.exp( 1j * phase_tilt )

	wf = Wavefront(aper)
	wf.wavelength = wavelength
	wf.total_power = 1
	for i, prop in enumerate( props ):
		img = prop(wf)
		plt.subplot(Ntilts,2,j*2+i+1)
		imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-5)
plt.show()