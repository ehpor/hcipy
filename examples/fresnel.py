'''
Test Fresnel propagator symmetries.
'''

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

N = 513
f = 10

pupil_grid = make_pupil_grid(N, 0.1)
aperture = circular_aperture(0.01)(pupil_grid)

for i in range(1000):
    prop = FresnelPropagator(pupil_grid, f*i)
    p = prop.get_monochromatic_propagator(1e-6)

    wf = Wavefront(aperture, 1e-6)
    wf2 = prop(wf)

    y = np.abs(p.fft.forward(wf.electric_field))
    print(y.grid.shape, y.grid.size)

    #plt.clf()
    #plt.plot(p.fft.output_grid.x, p.fft.output_grid.y, '+')
    #plt.show()

    #plt.clf()
    #imshow_field(np.abs(p.fft.forward(wf.electric_field)))
    #plt.show()

    plt.clf()
    imshow_field(wf2.intensity)
    plt.colorbar()
    plt.draw()
    plt.pause(0.001)