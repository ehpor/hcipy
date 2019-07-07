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

def test_fresnel_propagation_rectangular():
	for num_pix in [1024, 2048]:
		for distance in [0.3, 0.6, 1]:
			for wavelength in [500e-9, 700e-9]:
				for a, b in [[0.001, 0.001], [0.0015, 0.001]]:
					wavenumber = 2 * np.pi / wavelength

					pupil_grid = make_pupil_grid(num_pix, [32 * a, 32 * b])
					prop = FresnelPropagator(pupil_grid, distance)

					aperture = evaluate_supersampled(rectangular_aperture([2 * a, 2 * b]), pupil_grid, 1)

					img = prop(Wavefront(aperture, wavelength)).intensity

					w_x1 = np.sqrt(2 / (distance * wavelength)) * (pupil_grid.x - a)
					w_x2 = np.sqrt(2 / (distance * wavelength)) * (pupil_grid.x + a)
					w_y1 = np.sqrt(2 / (distance * wavelength)) * (pupil_grid.y - b)
					w_y2 = np.sqrt(2 / (distance * wavelength)) * (pupil_grid.y + b)
					
					fresnel = scipy.special.fresnel

					ssa_x1, csa_x1 = fresnel(w_x1)
					ssa_x2, csa_x2 = fresnel(w_x2)
					ssa_y1, csa_y1 = fresnel(w_y1)
					ssa_y2, csa_y2 = fresnel(w_y2)

					F_x1 = csa_x1 + 1j * ssa_x1
					F_x2 = csa_x2 + 1j * ssa_x2
					F_y1 = csa_y1 + 1j * ssa_y1
					F_y2 = csa_y2 + 1j * ssa_y2

					reference = (F_x2 - F_x1) * (F_y2 - F_y1)
					reference *= -2j * np.exp(1j * wavenumber * distance)
					reference *= np.exp(1j * wavenumber * (pupil_grid.x**2 + pupil_grid.y**2) / (2 * distance))
					reference = np.abs(Field(reference, pupil_grid))**2

					img = subsample_field(img, 8)
					reference = subsample_field(reference, 8)

					img /= np.sum(img)
					reference /= np.sum(reference)

					assert (np.abs(img - reference).max() / reference.max()) < 1e-2
