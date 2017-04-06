from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

pupil_grid = make_pupil_grid(512)
N = 9

zernike_basis = make_zernike_basis(N, 1, pupil_grid, 4)

for i,m in enumerate(zernike_basis):
    plt.subplot(3,3,i+1)
    imshow_field(zernike_basis[i], cmap='RdBu')
    plt.axis('off')
plt.show()

aperture = circular_aperture(1)(pupil_grid)

angles = np.arange(N) * (np.pi / N)
r = 30
positions = [np.array([np.sin(angle)*r, np.cos(angle)*r]) for angle in angles]

holo = make_multiplexed_cgh(pupil_grid, zernike_basis, positions, 1)
holo = np.sign(holo) * np.pi/2

imshow_field(holo)
plt.show()

aper = aperture * np.exp(0.2j * holo)

focal_grid = make_focal_grid(pupil_grid, 8, 45)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

wf = Wavefront(aper)
img = prop(wf)

imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-5)
plt.colorbar()
plt.show()