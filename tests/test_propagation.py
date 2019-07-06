import numpy as np
from hcipy import *
import scipy.special

def test_fraunhofer_propagation_circular():
	for num_pix in [512, 1024]:
		pupil_grid = make_pupil_grid(num_pix)
		focal_grid = make_focal_grid(pupil_grid, 16, 8)

		for diameter in [1, 0.7]:
			aperture = evaluate_supersampled(circular_aperture(diameter), pupil_grid, 8)

			for focal_length in [1, 0.8]:
				prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

				for wavelength in [1, 0.5]:
					wf = Wavefront(aperture, wavelength)
					img = prop(wf).electric_field
					img /= img[np.argmax(np.abs(img))]

					x = focal_grid.as_('polar').r * np.pi / wavelength * diameter / focal_length
					reference = 2 * scipy.special.jv(1, x) / x
					reference[focal_grid.closest_to((0, 0))] = 1

					if num_pix == 512:
						assert np.abs(img - reference).max() < 3e-5
					elif num_pix == 1024:
						assert np.abs(img - reference).max() < 1e-5
					else:
						assert False # This should never happen.
	
def test_fraunhofer_propagation_rectangular():
	for num_pix in [512, 1024]:
		pupil_grid = make_pupil_grid(num_pix)
		focal_grid = make_focal_grid(pupil_grid, 16, 8)

		for size in [[1,1], [0.75,1], [0.75,0.75]]:
			aperture = evaluate_supersampled(rectangular_aperture(size), pupil_grid, 8)

			for focal_length in [1, 1.3]:
				prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

				for wavelength in [1, 0.8]:
					wf = Wavefront(aperture, wavelength)
					img = prop(wf).electric_field
					img /= img[np.argmax(np.abs(img))]

					k_x, k_y = np.array(size) / wavelength / focal_length
					reference = (np.sinc(k_x * focal_grid.x) * np.sinc(k_y * focal_grid.y))

					if num_pix == 512:
						assert np.abs(img - reference).max() < 5e-5
					elif num_pix == 1024:
						assert np.abs(img - reference).max() < 2e-5
					else:
						assert False # This should never happen.