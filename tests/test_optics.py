import numpy as np 
from hcipy import *

def test_statistics_noisy_detector():
	N = 256
	grid = make_pupil_grid(N)

	test_wavefront = Field(np.ones(N**2), grid)
	
	#First we test photon noise, dark current noise and read noise.
	flat_field = 0
	dark_currents = np.logspace(1, 6, 6)
	read_noises = np.logspace(1, 6, 6)
	photon_noise = True

	for dc in dark_currents:
		for rn in read_noises: 
			#The test detector.
			test_detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=flat_field, dark_current_rate=dc, read_noise=rn)

			#The integration times we will test.
			integration_time = np.logspace(1,6,6)

			for t in integration_time:
				# integration
				test_detector.integrate(test_wavefront, t)

				# read out
				measurement = test_detector.read_out()

				# The std of the data by the detector.
				std_measurement = np.std(measurement)

				# The std that we expect given the input.
				expected_std = np.sqrt(test_wavefront[0] * t + rn**2 + dc * t)

				assert np.isclose(expected_std, std_measurement, rtol=2e-02, atol=1e-05)

	#Now we test the flat field separately.
	flat_fields = np.linspace(0, 1, 100)
	dark_current = 0
	read_noise = 0
	photon_noise = False

	for ff in flat_fields:
		#The test detector.
		test_detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=ff, dark_current_rate=dark_current, read_noise=read_noise)

		#The integration times we will test.
		integration_time = np.logspace(1, 6, 6)

		for t in integration_time:
			# integration
			test_detector.integrate(test_wavefront, t)

			# read out 
			measurement = test_detector.read_out()

			# The std of the data by the detector.
			std_measurement = np.std(measurement)

			# The std that we expect given the input.
			expected_std = ff * test_wavefront[0] * t 

			assert np.isclose(expected_std, std_measurement, rtol=2e-02, atol=1e-05)

def test_segmented_deformable_mirror():
	num_pix = 256
	grid = make_pupil_grid(num_pix)

	for num_rings in [2, 4, 5]:
		num_segments_expected = 3 * (num_rings + 1) * num_rings + 1

		segment_positions = make_hexagonal_grid(0.5 / num_rings * np.sqrt(3) / 2, num_rings)
		aperture, segments = make_segmented_aperture(hexagonal_aperture(0.5 / num_rings - 0.003, np.pi / 2), segment_positions, return_segments=True)

		aperture = evaluate_supersampled(aperture, grid, 2)
		segments = evaluate_supersampled(segments, grid, 2)

		# Check number of generated segments
		assert len(segments) == num_segments_expected

		segmented_mirror = SegmentedDeformableMirror(segments)

		# Mirror should start out flattened.
		assert np.std(segmented_mirror.surface) < 1e-12

		for i in np.random.randint(0, num_segments_expected, size=10):
			piston = np.random.randn(1)

			while np.abs(piston) < 1e-5:
				piston = np.random.randn(1)

			segmented_mirror.set_segment_actuators(i, piston, 0, 0)

			# Mirror should have the correct piston
			assert np.abs(segmented_mirror.get_segment_actuators(i)[0] - piston) / piston < 1e-5
			assert np.abs(np.ptp(segmented_mirror.surface) - piston) / piston < 1e-5

			tip, tilt = np.random.randn(2)

			while np.abs(tip) < 1e-5:
				tip = np.random.randn(1)

			while np.abs(tilt) < 1e-5:
				tilt = np.random.randn(1)

			segmented_mirror.set_segment_actuators(i, 0, tip, tilt)

			# Mirror should be distorted with tip and tilt on a segment
			assert np.std(segmented_mirror.surface > 1e-12)

			# Return segment to zero position
			segmented_mirror.set_segment_actuators(i, 0, 0, 0)
