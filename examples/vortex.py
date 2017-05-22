from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

pupil_grid = make_pupil_grid(256)
focal_grid = make_focal_grid(pupil_grid, 16, 32)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

vortex = VortexCoronagraph(pupil_grid, 1, 2, 'GS')

aperture = circular_aperture(1)(pupil_grid)
wf = Wavefront(aperture)
wf2 = vortex(wf)
#wf2 = wf
wf2.electric_field *= aperture
wf3 = prop(wf2)

imshow_field(np.log10(wf2.intensity))
plt.colorbar()
plt.show()

imshow_field(np.log10(wf3.intensity))
plt.colorbar()
plt.show()