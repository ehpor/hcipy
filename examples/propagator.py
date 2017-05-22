from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

pupil_grid = make_pupil_grid(512)
aperture = circular_aperture(1)

A = aperture(pupil_grid)

imshow_field(A)
plt.show()

focal_grid = make_focal_grid(pupil_grid, 8, 100, wavelength=1E-6)

fresnel = FresnelPropagator(pupil_grid, 500)
fraunhofer = FraunhoferPropagator(pupil_grid, focal_grid, wavelength_0=1E-6)

wf = Wavefront(A, wavelength=1E-6)
wf.total_power = 1

img = fresnel(wf)
print(wf.total_power, img.total_power)

img = fraunhofer(wf)
print(wf.total_power, img.total_power)

imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-5)
plt.show()