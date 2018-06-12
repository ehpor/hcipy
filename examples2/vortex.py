from hcipy import *
import matplotlib.pyplot as plt
import numpy as np
import time

pupil_grid = make_pupil_grid(128)
focal_grid = make_focal_grid(pupil_grid, 8, 16)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

vortex = VortexCoronagraph(pupil_grid, charge=2, scalings=2)

for i, m in enumerate(vortex.focal_masks):
	plt.subplot(3,3,i+1)
	imshow_field(m)
plt.show()

pupil_grid_large = make_pupil_grid(pupil_grid.shape * 8)
aperture = circular_aperture(1)(pupil_grid_large)
aperture = aperture.reshape((pupil_grid.shape[1],8,pupil_grid.shape[0],8)).mean(axis=(1,3))
aperture = Field((aperture.ravel()), pupil_grid)

imshow_field(aperture, cmap='gray')
plt.show()

wf = Wavefront(aperture)
lyot = vortex(wf)

N = 10
start = time.time()
for i in range(N):
	lyot = vortex(wf)
end = time.time()

print('time per vortex propagation:', (end - start) / N, 'sec')

imshow_field(np.log10(lyot.intensity))
plt.show()

lyot.electric_field *= circular_aperture(0.99)(pupil_grid)

img = prop(lyot).intensity
img_ref = prop(Wavefront(aperture)).intensity

imshow_field(np.log10(img / img_ref.max()), vmin=-12)
plt.colorbar()
plt.show()