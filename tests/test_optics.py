import numpy as np 
from hcipy import *

def test_statistics_noisy_detector():
	N = 256
	grid = make_pupil_grid(N)

	field = Field(np.ones(N**2), grid)
	
	#First we test photon noise, dark current noise and read noise.
	flat_field = 0
	dark_currents = np.logspace(1, 6, 6)
	read_noises = np.logspace(1, 6, 6)
	photon_noise = True

	for dc in dark_currents:
		for rn in read_noises: 
			#The test detector.
			detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=flat_field, dark_current_rate=dc, read_noise=rn)

			#The integration times we will test.
			integration_time = np.logspace(1,6,6)

			for t in integration_time:
				# integration
				detector.integrate(field, t)

				# read out
				measurement = detector.read_out()

				# The std of the data by the detector.
				std_measurement = np.std(measurement)

				# The std that we expect given the input.
				expected_std = np.sqrt(field[0] * t + rn**2 + dc * t)

				assert np.isclose(expected_std, std_measurement, rtol=2e-02, atol=1e-05)

	#Now we test the flat field separately.
	flat_fields = np.linspace(0, 1, 100)
	dark_current = 0
	read_noise = 0
	photon_noise = False

	for ff in flat_fields:
		#The test detector.
		detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=ff, dark_current_rate=dark_current, read_noise=read_noise)

		#The integration times we will test.
		integration_time = np.logspace(1, 6, 6)

		for t in integration_time:
			# integration
			detector.integrate(field, t)

			# read out 
			measurement = detector.read_out()

			# The std of the data by the detector.
			std_measurement = np.std(measurement)

			# The std that we expect given the input.
			expected_std = ff * field[0] * t 

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

def test_wavefront_stokes():
	N = 4
	grid = make_pupil_grid(N)

	stokes = np.random.uniform(-1,1,4)
	stokes[0] = np.abs(stokes[0])+np.linalg.norm(stokes[1:])

	amplitude = np.random.uniform(0, 1, (2,2,N**2))
	phase = np.random.uniform(0, 2 * np.pi, (2,2,N**2))

	jones_field = Field(amplitude * np.exp(1j * phase), grid)

	determinants = field_determinant(jones_field)
	jones_field[:, :, determinants > 1] *= np.random.uniform(0, 1, np.sum(determinants > 1)) / determinants[determinants > 1]

	jones_element = JonesMatrixOpticalElement(jones_field)

	mueller_field = jones_to_mueller(jones_field)

	field = Field(np.ones(N**2), grid)
	stokes_wavefront = Wavefront(field, stokes_vector=stokes)

	jones_element_forward = jones_element.forward(stokes_wavefront)
	mueller_forward = field_dot(mueller_field, stokes)

	assert np.allclose(jones_element_forward.I, mueller_forward[0])
	assert np.allclose(jones_element_forward.Q, mueller_forward[1])
	assert np.allclose(jones_element_forward.U, mueller_forward[2])
	assert np.allclose(jones_element_forward.V, mueller_forward[3])

def test_degree_and_angle_of_polarization():
	grid = make_pupil_grid(16)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0, 0, 0])
	assert np.allclose(wf.degree_of_polarization, 0)
	assert np.allclose(wf.degree_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, 0)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 1, 0, 0])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 1)
	assert np.allclose(wf.angle_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, 0)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0.5, 0, 0])
	assert np.allclose(wf.degree_of_polarization, 0.5)
	assert np.allclose(wf.degree_of_linear_polarization, 0.5)
	assert np.allclose(wf.angle_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, 0)

	wf = Wavefront(grid.ones(), stokes_vector=[1, -1, 0, 0])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 1)
	assert np.allclose(wf.angle_of_linear_polarization, np.pi / 2)
	assert np.allclose(wf.degree_of_circular_polarization, 0)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0, 1, 0])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 1)
	assert np.allclose(wf.angle_of_linear_polarization, np.pi / 4)
	assert np.allclose(wf.degree_of_circular_polarization, 0)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0, 0, 1])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, 1)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0, 0, 0.5])
	assert np.allclose(wf.degree_of_polarization, 0.5)
	assert np.allclose(wf.degree_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, 0.5)

	wf = Wavefront(grid.ones(), stokes_vector=[1, 0, 0, -1])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 0)
	assert np.allclose(wf.degree_of_circular_polarization, -1)

	wf = Wavefront(grid.ones(), stokes_vector=[2, np.sqrt(2), 0, np.sqrt(2)])
	assert np.allclose(wf.degree_of_polarization, 1)
	assert np.allclose(wf.degree_of_linear_polarization, 1 / np.sqrt(2))
	assert np.allclose(wf.degree_of_circular_polarization, 1 / np.sqrt(2))
