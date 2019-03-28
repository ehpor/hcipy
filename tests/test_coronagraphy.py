from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def test_vortex_coronagraph():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(pupil_grid, 4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = circular_aperture(1)
	aperture = evaluate_supersampled(aperture, pupil_grid, 8)

	lyot = circular_aperture(0.99)
	lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

	for charge in [2,4,6,8]:
		vortex = VortexCoronagraph(pupil_grid, charge, levels=6)

		wf = Wavefront(aperture)
		wf.total_power = 1

		img_ref = prop(wf)

		wf = vortex(wf)
		wf.electric_field *= lyot
		img = prop(wf)

		assert img.total_power < 1e-6
		assert img.intensity.max() / img_ref.intensity.max() < 1e-8

def test_ravc():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(pupil_grid, 4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	for co in [0.1, 0.2, 0.3]:
		aperture = lambda grid: circular_aperture(1)(grid) - circular_aperture(co)(grid)
		aperture = evaluate_supersampled(aperture, pupil_grid, 16)

		aper = aperture > 1e-5

		for charge in [2,4]:
			pupil, lyot = make_ravc_masks(co, charge, lyot_undersize=0.02)

			pupil = evaluate_supersampled(pupil, pupil_grid, 4)
			lyot = evaluate_supersampled(lyot, pupil_grid, 4)

			vortex = VortexCoronagraph(pupil_grid, charge, levels=6)

			wf = Wavefront(aper)
			wf.total_power = 1
			img_ref = prop(wf)

			wf = Wavefront(aper)
			wf.total_power = 1
			wf.electric_field *= pupil
			wf = vortex(wf)
			wf.electric_field *= lyot
			img = prop(wf)

			assert img.total_power < 1e-6
			assert img.intensity.max() / img_ref.intensity.max() < 1e-8

			pupil, lyot = make_ravc_masks(co, charge)
			pupil = evaluate_supersampled(pupil, pupil_grid, 4)
			lyot = evaluate_supersampled(lyot, pupil_grid, 4)

			transmission = ((pupil * lyot)**2).sum() / (aperture**2).sum()
			transmission_theoretical = get_ravc_planet_transmission(co, charge)

			assert abs(transmission - transmission_theoretical) < 0.01

def test_app_keller():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(pupil_grid, 4, 32)
	propagator = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = evaluate_supersampled(circular_aperture(1), pupil_grid, 8)
	wavefront = Wavefront(aperture)
	wavefront.total_power = 1

	# reference PSF without APP
	img_ref = propagator.forward(wavefront)

	# small rectangular dark zone with 1e-7 contrast
	mask = rectangular_aperture(size=(6,2), center=(9,0))(focal_grid)
	contrast = 1 - mask + 1e-7

	# APP with aggressive acceleration
	app = generate_app_keller(wavefront, propagator, contrast,
		num_iterations = 150, beta = 0.98)
	img = propagator.forward(app)

	assert img.intensity.max() / img_ref.intensity.max() > 0.92 # Strehl
	assert np.mean(img.intensity * mask) / np.mean(mask) < 1.5e-8 # contrast
