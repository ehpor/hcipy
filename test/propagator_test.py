from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

pupil_grid = make_pupil_grid(2048, 0.2)
aperture = circular_aperture(0.1)

A = aperture(pupil_grid)

imshow_field(A)
plt.show()

propagator = FresnelPropagator(pupil_grid, 5)

wf = Wavefront(A, 500e-9)
img = propagator(wf)

prop_mono = propagator.get_monochromatic_propagator(500e-9)

imshow_field(np.angle(prop_mono.transfer_function), prop_mono.fft.output_grid)
plt.show()

imshow_field(img.intensity / img.intensity.max())
plt.show()