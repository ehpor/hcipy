from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

pupil_grid = make_pupil_grid(512)

aperture = circular_aperture(1)(pupil_grid)

wf = Wavefront(aperture)
wf.total_power = 1

wf.electric_field *= np.exp(0.2j * np.cos(10 * 2*np.pi * pupil_grid.x))

plt.subplot(1,2,1)
imshow_field(wf.amplitude)
plt.colorbar()
plt.subplot(1,2,2)
imshow_field(wf.phase, cmap='RdBu')
plt.colorbar()
plt.show()

focal_grid = make_focal_grid(pupil_grid, 8, 20)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

img = prop(wf)
imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-5)
plt.colorbar()
plt.show()