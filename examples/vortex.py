from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

pupil_grid = make_pupil_grid(256)
focal_grid = make_focal_grid(pupil_grid, 16, 32)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

vortex = VortexCoronagraph(pupil_grid, 1, 2, 'gs')

plt.ion()
for tilt in np.linspace(0,2,50):
    aperture = circular_aperture(1)(pupil_grid)
    wf = Wavefront(aperture)
    wf.electric_field *= np.exp(2j * np.pi * pupil_grid.x * tilt)
    wf2 = vortex(wf)
    #wf2 = wf
    wf3 = wf2.copy()
    #wf2 = wf
    wf3.electric_field *= aperture
    wf4 = prop(wf3)

    #imshow_field(np.log10(wf2.intensity))
    #plt.colorbar()
    #plt.show()

    plt.clf()
    plt.title('tilt=%.2f' % tilt)
    imshow_field(np.log10(wf4.intensity))
    plt.colorbar()
    plt.draw()
    plt.pause(0.0001)
plt.ioff()