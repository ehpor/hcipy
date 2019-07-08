from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# pupil plane grid with 256 by 256 pixels
pupil_grid = make_pupil_grid(256)

# focal plane grid with 4 samples per lambda/D out to 32 lambda/D
focal_grid = make_focal_grid(pupil_grid, 4, 32)

# propagator from the pupil grid to the focal plane grid
propagator = FraunhoferPropagator(pupil_grid, focal_grid)

# unobstructed, circular aperture in pupil
aperture = evaluate_supersampled(circular_aperture(1), pupil_grid, 8)

# wavefront corresponding to aperture
wavefront = Wavefront(aperture)
wavefront.total_power = 1

plt.subplot(2,2,1)
plt.title('aperture')
imshow_field(wavefront.amplitude)
plt.colorbar()

# reference PSF without APP
img_ref = propagator.forward(wavefront)
plt.subplot(2,2,2)
plt.title('PSF without APP')
imshow_field(np.log10(img_ref.intensity / img_ref.intensity.max()), vmin=-7)
plt.colorbar()

# contrast mask as a field in the focal plane, all sizes and centers in
# units of lambda/D
mask = rectangular_aperture(size=(6,2), center=(9,0))(focal_grid)
contrast = 1 - mask + 1e-7

# maximum number of iterations
num_iterations = 500

# acceleration parameter
beta = 0.98

# make APP using accelerated iterative approach by C.U.Keller
app = generate_app_keller(wavefront, propagator, contrast,
	num_iterations, beta)

# calculate PSF with APP
img = propagator.forward(app)
plt.subplot(2,2,3)
plt.title('APP phase')
imshow_field(app.phase)
plt.colorbar()

plt.subplot(2,2,4)
plt.title('PSF with APP')
imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-7)
plt.colorbar()

print('Strehl ratio     =', img.intensity.max() / img_ref.intensity.max())
print('average contrast =', np.mean(img.intensity * mask) / np.mean(mask))

plt.show()
