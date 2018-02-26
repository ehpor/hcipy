from hcipy import *
from matplotlib import pyplot as plt
import numpy as np

def test_pupil_plane_wavefront_sensors():
	# Create the input grid
	D = 1
	pupil_grid = make_pupil_grid(128, D)
	aperture = circular_aperture(D)

	# Make zernike basis
	num_zernike = 3
	zernike_basis = make_zernike_basis(num_zernike, D, pupil_grid, 2)

	# Create wavefront sensors
	pyramid_wfs = PyramidWavefrontSensorOptics(pupil_grid)
	zernike_wfs = ZernikeWavefrontSensorOptics(pupil_grid, q=32, num_airy=20)
	god_wfs = gODWavefrontSensorOptics(40, 1/3, pupil_grid, pupil_separation=1.5, q=16, num_airy=40)
	pod_wfs = PolgODWavefrontSensorOptics(40, 1/3, pupil_grid, pupil_separation=1.5, q=16, num_airy=40)

	wavefront_sensors = (pyramid_wfs, zernike_wfs, god_wfs, pod_wfs)
	num_wfs = len(wavefront_sensors)

	# Create reference images
	wf = Wavefront(aperture(pupil_grid))
	wf.total_power = 1
	refs = [wfs.forward(wf).intensity for wfs in wavefront_sensors]
	eff = [wfs.forward(wf).total_power for wfs in wavefront_sensors]
	print("Total efficiency : ", eff)

	for i in range(num_wfs):
		plt.subplot(2,2,i+1)
		imshow_field(refs[i], wavefront_sensors[i].output_grid)
	plt.show()

	for i, zern in enumerate(zernike_basis):
		# create input
		wf = Wavefront(aperture(pupil_grid) * np.exp(1j * 0.01 * zern))
		wf.total_power = 1

		# Measure the response of all wavefront sensors
		for j, wfs in enumerate(wavefront_sensors):
			wf_out = wfs.forward(wf)
			print(wf_out.total_power)
			plt.subplot(num_wfs, num_zernike, i+1 + num_zernike*j)
			imshow_field(wf_out.intensity-refs[j])

	plt.show()



if __name__ == "__main__":
	test_pupil_plane_wavefront_sensors()