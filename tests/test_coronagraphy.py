from hcipy import *
import numpy as np

def test_vortex_coronagraph():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = circular_aperture(1)
	aperture = evaluate_supersampled(aperture, pupil_grid, 8)

	lyot = circular_aperture(0.99)
	lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

	for charge in [2, 4, 6, 8]:
		vortex = VortexCoronagraph(pupil_grid, charge)

		wf = Wavefront(aperture)
		wf.total_power = 1

		img_ref = prop(wf)

		wf = vortex(wf)
		wf.electric_field *= lyot
		img = prop(wf)

		assert img.total_power < 1e-6
		assert img.intensity.max() / img_ref.intensity.max() < 1e-8

def test_vector_vortex_coronagraph():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = circular_aperture(1)
	aperture = evaluate_supersampled(aperture, pupil_grid, 8)

	lyot = circular_aperture(0.99)
	lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

	for charge in [2, 4, 6, 8]:
		vortex = VectorVortexCoronagraph(charge)

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
	focal_grid = make_focal_grid(4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	for co in [0.1, 0.2, 0.3]:
		aperture = lambda grid: circular_aperture(1)(grid) - circular_aperture(co)(grid)
		aperture = evaluate_supersampled(aperture, pupil_grid, 16)

		aper = aperture > 1e-5

		for charge in [2, 4]:
			pupil, lyot = make_ravc_masks(co, charge, lyot_undersize=0.02)

			pupil = evaluate_supersampled(pupil, pupil_grid, 4)
			lyot = evaluate_supersampled(lyot, pupil_grid, 4)

			vortex = VortexCoronagraph(pupil_grid, charge)

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
	focal_grid = make_focal_grid(4, 32)
	propagator = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = evaluate_supersampled(circular_aperture(1), pupil_grid, 8)
	wavefront = Wavefront(aperture)
	wavefront.total_power = 1

	# reference PSF without APP
	img_ref = propagator.forward(wavefront)

	# small rectangular dark zone with 1e-7 contrast
	mask = rectangular_aperture(size=(6, 2), center=(9, 0))(focal_grid)
	contrast = 1 - mask + 1e-7

	# APP with aggressive acceleration
	app = generate_app_keller(wavefront, propagator, contrast, num_iterations=150, beta=0.98)
	img = propagator.forward(app)

	assert np.abs(img.intensity.max() / img_ref.intensity.max() - 0.947) < 0.01  # Strehl
	assert np.mean(img.intensity * mask) / np.mean(mask) < 1.6e-8  # contrast

def test_perfect_coronagraph():
	pupil_grid = make_pupil_grid(256)
	aperture = circular_aperture(1)(pupil_grid)

	tilts = np.logspace(-3, -1, 51)

	for order in [2, 4, 6, 8]:
		coro = PerfectCoronagraph(aperture, order)

		# Test suppression for on-axis point source
		wf = Wavefront(aperture)
		wf.total_power = 1
		assert coro(wf).total_power < 1e-10

		# Test suppression off-axis
		coronagraph_leakage = []
		for tilt in tilts:
			leakage = coro(Wavefront(aperture * np.exp(2j * np.pi * pupil_grid.x * tilt))).total_power
			coronagraph_leakage.append(leakage)

		y = np.log10(coronagraph_leakage)
		x = np.log10(tilts)
		n = len(x)

		# Do a linear fit on the log-log data to get the power-law coefficient
		beta = ((x * y).sum() - x.sum() * y.sum() / n) / ((x * x).sum() - x.sum()**2 / n)
		assert np.abs(beta - order) / order < 1e-3

def test_lyot_coronagraph():
	pupil_grid = make_pupil_grid(128, 1.1)
	aperture = evaluate_supersampled(circular_aperture(1.0), pupil_grid, 8)
	lyot_stop = evaluate_supersampled(circular_aperture(0.95), pupil_grid, 8)

	# Coronagraph 1 with the default internal focal length
	fpm_grid = make_focal_grid(q=32, num_airy=3)
	fpm = 1 - evaluate_supersampled(circular_aperture(5), fpm_grid, 8)
	cor = LyotCoronagraph(pupil_grid, fpm, lyot_stop)

	# Coronagraph 2 with a large focal length
	focal_length = 10.0
	fpm_grid2 = make_focal_grid(q=32, num_airy=3, spatial_resolution=focal_length)
	fpm2 = 1 - evaluate_supersampled(circular_aperture(5 * focal_length), fpm_grid2, 8)
	cor2 = LyotCoronagraph(pupil_grid, fpm2, lyot_stop, focal_length=focal_length)

	# The grid on which the performance is evaluated
	focal_grid = make_focal_grid(q=3, num_airy=25)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	wf = Wavefront(aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()
	wf_foc = prop(cor(wf))
	wf_foc2 = prop(cor2(wf))

	# Checks performance of the coronagraph and if the focal length does not introduce artifacts
	assert (wf_foc.power.max() / norm) < 5e-3
	np.testing.assert_allclose(wf_foc.power, wf_foc2.power)


def test_knife_edge_coronagraph():
	grid = make_pupil_grid(64, 1.1)
	aperture = circular_aperture(1)(grid)

	focal_grid = make_focal_grid(q=5, num_airy=3)

	prop = FraunhoferPropagator(grid, focal_grid)
	lyot_stop = circular_aperture(0.95)(grid)

	wf = Wavefront(aperture)
	wf.total_power = 1.0
	norm = prop(wf).power.max()

	directions = ['+x', '-x', '+y', '-y']
	for direction in directions:
		knife_left = KnifeEdgeLyotCoronagraph(grid, direction=direction, apodizer=None, lyot_stop=lyot_stop)
		wf_cor = prop(knife_left(wf))

		assert (wf_cor.power.max() / norm) < 0.25

	directions = ['+x', '-x', '+y', '-y']
	knife_edge_shifts = [-1.0 * grid.x, 1.0 * grid.x, -1.0 * grid.y, 1.0 * grid.y]
	for shift, direction in zip(knife_edge_shifts, directions):
		pre_apodizer = np.exp(1j * 2 * np.pi * shift)
		knife_left = KnifeEdgeLyotCoronagraph(grid, direction=direction, apodizer=pre_apodizer, lyot_stop=lyot_stop * np.conj(pre_apodizer))
		wf_cor = prop(knife_left(wf))

		assert (wf_cor.power.max() / norm) < 1e-2
