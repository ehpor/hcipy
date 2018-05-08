from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def test_vortex_coronagraph():
	pupil_grid = make_pupil_grid(256)
	focal_grid = make_focal_grid(pupil_grid, 4, 32)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	aperture = circular_aperture(1)
	aperture = evaluate_supersampled(aperture, pupil_grid, 16)

	lyot = circular_aperture(0.99)
	lyot = evaluate_supersampled(lyot, pupil_grid, 16) > 1 - 1e-5

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
	pass