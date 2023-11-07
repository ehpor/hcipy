import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
import pytest
import os

def test_agnostic_apodizer():
    aperture_achromatic = make_circular_aperture(1)
    aperture_chromatic = lambda input_grid, wavelength: make_circular_aperture(wavelength)(input_grid)

    apod_achromatic = Apodizer(aperture_achromatic)
    apod_chromatic = Apodizer(aperture_chromatic)

    phase_chromatic = lambda input_grid, wavelength: zernike(2, 0)(input_grid) * wavelength
    apod_phase_chromatic = PhaseApodizer(phase_chromatic)

    phase_achromatic = zernike(2, 0)
    apod_phase_achromatic = PhaseApodizer(phase_achromatic)

    filter_curve = lambda wavelength: np.sqrt(wavelength)
    apod_filter = Apodizer(filter_curve)

    for wl in [0.2, 0.4, 0.7]:
        for num_pix in [32, 64, 128]:
            for diameter in [0.5, 1, 2]:
                pupil_grid = make_pupil_grid(num_pix, diameter)
                wf = Wavefront(pupil_grid.ones(), wl)

                pup = apod_achromatic(wf).electric_field
                assert np.allclose(pup, aperture_achromatic(pupil_grid))

                pup = apod_chromatic(wf).electric_field
                assert np.allclose(pup, aperture_chromatic(pupil_grid, wl))

                pup = apod_phase_achromatic(wf).electric_field
                assert np.allclose(pup, np.exp(1j * phase_achromatic(pupil_grid)))

                pup = apod_phase_chromatic(wf).electric_field
                assert np.allclose(pup, np.exp(1j * phase_chromatic(pupil_grid, wl)))

                pup = apod_filter(wf).electric_field
                assert np.allclose(pup, filter_curve(wl))

def test_statistics_noisy_detector():
    N = 256
    grid = make_pupil_grid(N)

    field = Field(np.ones(N**2), grid)

    # First we test photon noise, dark current noise and read noise.
    flat_field = 0
    dark_currents = np.logspace(1, 6, 6)
    read_noises = np.logspace(1, 6, 6)
    photon_noise = True

    for dc in dark_currents:
        for rn in read_noises:
            # The test detector.
            detector = NoisyDetector(detector_grid=grid, include_photon_noise=photon_noise, flat_field=flat_field, dark_current_rate=dc, read_noise=rn)

            # The integration times we will test.
            integration_time = np.logspace(1, 6, 6)

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

    # Now we test the flat field separately.
    flat_fields = np.linspace(0, 1, 100)
    dark_current = 0
    read_noise = 0
    photon_noise = False

    for ff in flat_fields:
        detector = NoisyDetector(detector_grid=grid, include_photon_noise=photon_noise, flat_field=ff, dark_current_rate=dark_current, read_noise=read_noise)

        integration_times = np.logspace(1, 6, 6)

        for t in integration_times:
            # integration and read out
            detector.integrate(field, t)
            measurement = detector.read_out()

            # The std of the data by the detector.
            std_measurement = np.std(measurement)

            # The std that we expect given the input.
            expected_std = ff * field[0] * t

            assert np.isclose(expected_std, std_measurement, rtol=2e-02, atol=1e-05)

    aperture = make_circular_aperture(1)(grid)
    subsamplings = [1, 2, 4, 8]

    for subsampling in subsamplings:
        grid_subsampled = make_subsampled_grid(grid, subsampling)
        aperture_subsampled = subsample_field(aperture, subsampling, grid_subsampled, statistic='sum')

        # the detector with the new subsampling factor
        detector = NoisyDetector(detector_grid=grid_subsampled, include_photon_noise=False, subsampling=subsampling)

        # integrating and reading out
        detector.integrate(aperture, dt=1)
        detector_image = detector.read_out()

        # testing if the image from the detector matches the subsampled aperture
        assert np.allclose(aperture_subsampled, detector_image)

def test_glass_catalogue():
    bk7 = get_refractive_index('N-BK7')
    assert np.allclose(bk7(500e-9), 1.5214144761028994)

    with pytest.raises(ValueError) as exception_info:
        get_refractive_index('N-Bk7')
    assert 'Did you mean' in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        get_refractive_index('N-Ba7')
    assert 'Did you mean' not in str(exception_info.value)

def test_deformable_mirror():
    num_pix = 256
    x_tilt = np.radians(15)
    y_tilt = np.radians(10)
    z_tilt = np.radians(20)
    grid = make_pupil_grid(num_pix, 1.5)

    functions = [make_gaussian_influence_functions, make_xinetics_influence_functions]

    for num_actuators_across_pupil in [12, 16]:
        actuator_spacing = 1 / num_actuators_across_pupil
        num_actuators = num_actuators_across_pupil**2

        # Check if the actuator spacing of an unrotated DM is correct
        actuator_positions = make_actuator_positions(num_actuators_across_pupil, actuator_spacing)
        assert np.allclose(actuator_positions.delta, actuator_spacing)

        actuator_positions = make_actuator_positions(num_actuators_across_pupil, actuator_spacing, x_tilt=x_tilt, y_tilt=y_tilt, z_tilt=z_tilt)

        for func in functions:
            influence_functions = func(grid, num_actuators_across_pupil, actuator_spacing, x_tilt=x_tilt, y_tilt=y_tilt, z_tilt=z_tilt)

            deformable_mirror = DeformableMirror(influence_functions)

            # Check number of generated influence functions
            assert len(influence_functions) == num_actuators
            assert deformable_mirror.num_actuators == num_actuators

            # Check that influence functions are sparse
            assert influence_functions.is_sparse

            # Check if the centroids for each influence function are close to the predicted position.
            for act, p in zip(influence_functions, actuator_positions.points):
                x_pos = (act * grid.x).sum() / act.sum()
                y_pos = (act * grid.y).sum() / act.sum()

                assert np.allclose([x_pos, y_pos], p, atol=0.02)

            # Mirror should start out flattened
            assert np.std(deformable_mirror.surface) < 1e-12

            for i in np.random.randint(0, num_actuators, size=10):
                deformable_mirror.actuators[i] = np.random.randn(1)[0]

                # Mirror should not be flat with poked actuator
                assert np.std(deformable_mirror.surface) > 1e-12

                deformable_mirror.actuators[i] = 0

            # Mirror should be flat again
            assert np.std(deformable_mirror.surface) < 1e-12

            deformable_mirror.random(0.001)

            # Mirror should have random actuator pokes
            assert np.std(deformable_mirror.surface) > 1e-12

            wf = Wavefront(grid.ones())
            wf_out = deformable_mirror.forward(wf)

            # Check that the correct phase is applied
            assert np.allclose(wf_out.phase, deformable_mirror.phase_for(1))

            wf_in = deformable_mirror.backward(wf_out)

            # Check that the deformable mirror is phase only
            assert np.allclose(wf_in.electric_field, wf.electric_field)

            # Check OPD
            assert np.allclose(deformable_mirror.opd, 2 * deformable_mirror.surface)

            deformable_mirror.flatten()

            # Check that the deformable mirror is flat again
            assert np.std(deformable_mirror.surface) < 1e-12

def test_segmented_deformable_mirror():
    num_pix = 256
    grid = make_pupil_grid(num_pix)

    for num_rings in [2, 4, 5]:
        num_segments_expected = 3 * (num_rings + 1) * num_rings + 1

        segment_positions = make_hexagonal_grid(0.5 / num_rings * np.sqrt(3) / 2, num_rings)
        aperture, segments = make_segmented_aperture(make_hexagonal_aperture(0.5 / num_rings - 0.003, np.pi / 2), segment_positions, return_segments=True)

        aperture = evaluate_supersampled(aperture, grid, 2)
        segments = evaluate_supersampled(segments, grid, 2)

        # Check number of generated segments
        assert len(segments) == num_segments_expected

        segmented_mirror = SegmentedDeformableMirror(segments)

        # Mirror should start out flattened.
        assert np.std(segmented_mirror.surface) < 1e-12

        for i in np.random.randint(0, num_segments_expected, size=10):
            piston = np.random.randn(1)[0]

            while np.abs(piston) < 1e-5:
                piston = np.random.randn(1)[0]

            segmented_mirror.set_segment_actuators(i, piston, 0, 0)

            # Mirror should have the correct piston
            assert np.abs(segmented_mirror.get_segment_actuators(i)[0] - piston) / piston < 1e-5
            assert np.abs(np.ptp(segmented_mirror.surface) - piston) / piston < 1e-5

            tip, tilt = np.random.randn(2)

            while np.abs(tip) < 1e-5:
                tip = np.random.randn(1)[0]

            while np.abs(tilt) < 1e-5:
                tilt = np.random.randn(1)[0]

            segmented_mirror.set_segment_actuators(i, 0, tip, tilt)

            # Mirror should be distorted with tip and tilt on a segment
            assert np.std(segmented_mirror.surface > 1e-12)

            # Return segment to zero position
            segmented_mirror.set_segment_actuators(i, 0, 0, 0)

        # Run labeling of segments
        imshow_field(aperture)
        label_actuator_centroid_positions(segments)
        plt.clf()

def test_wavefront_stokes():
    N = 4
    grid = make_pupil_grid(N)

    stokes = np.random.uniform(-1, 1, 4)
    stokes[0] = np.abs(stokes[0]) + np.linalg.norm(stokes[1:])

    amplitude = np.random.uniform(0, 1, (2, 2, N**2))
    phase = np.random.uniform(0, 2 * np.pi, (2, 2, N**2))

    jones_field = Field(amplitude * np.exp(1j * phase), grid)

    determinants = field_determinant(jones_field)
    jones_field[:, :, determinants > 1] *= np.random.uniform(0, 1, np.sum(determinants > 1)) / determinants[determinants > 1]

    jones_element = JonesMatrixOpticalElement(jones_field)

    mueller_field = jones_to_mueller(jones_field)

    field = Field(np.ones(N**2), grid)
    stokes_wavefront = Wavefront(field, input_stokes_vector=stokes)

    jones_element_forward = jones_element.forward(stokes_wavefront)
    mueller_forward = field_dot(mueller_field, stokes)

    assert np.allclose(jones_element_forward.I, mueller_forward[0])
    assert np.allclose(jones_element_forward.Q, mueller_forward[1])
    assert np.allclose(jones_element_forward.U, mueller_forward[2])
    assert np.allclose(jones_element_forward.V, mueller_forward[3])

def test_degree_and_angle_of_polarization():
    grid = make_pupil_grid(16)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0, 0, 0])
    assert np.allclose(wf.degree_of_polarization, 0)
    assert np.allclose(wf.degree_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, 0)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 1, 0, 0])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 1)
    assert np.allclose(wf.angle_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, 0)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0.5, 0, 0])
    assert np.allclose(wf.degree_of_polarization, 0.5)
    assert np.allclose(wf.degree_of_linear_polarization, 0.5)
    assert np.allclose(wf.angle_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, 0)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, -1, 0, 0])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 1)
    assert np.allclose(wf.angle_of_linear_polarization, np.pi / 2)
    assert np.allclose(wf.degree_of_circular_polarization, 0)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0, 1, 0])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 1)
    assert np.allclose(wf.angle_of_linear_polarization, np.pi / 4)
    assert np.allclose(wf.degree_of_circular_polarization, 0)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0, 0, 1])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, 1)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0, 0, 0.5])
    assert np.allclose(wf.degree_of_polarization, 0.5)
    assert np.allclose(wf.degree_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, 0.5)

    wf = Wavefront(grid.ones(), input_stokes_vector=[1, 0, 0, -1])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 0)
    assert np.allclose(wf.degree_of_circular_polarization, -1)

    wf = Wavefront(grid.ones(), input_stokes_vector=[2, np.sqrt(2), 0, np.sqrt(2)])
    assert np.allclose(wf.degree_of_polarization, 1)
    assert np.allclose(wf.degree_of_linear_polarization, 1 / np.sqrt(2))
    assert np.allclose(wf.degree_of_circular_polarization, 1 / np.sqrt(2))

def mueller_matrix_for_general_linear_retarder(theta, delta):
    '''Analytic expression Mueller matrix linear retarder.

    Parameters
    ----------
    theta : scaler
        rotation angle optic in radians
    delta : scalar
        retardance optic in radians
    '''
    retarder = np.zeros((4, 4))

    retarder[0, 0] = 1

    retarder[1, 1] = np.cos(2 * theta)**2 + np.sin(2 * theta)**2 * np.cos(delta)
    retarder[1, 2] = np.cos(2 * theta) * np.sin(2 * theta) * (1 - np.cos(delta))
    retarder[1, 3] = np.sin(2 * theta) * np.sin(delta)

    retarder[2, 1] = np.cos(2 * theta) * np.sin(2 * theta) * (1 - np.cos(delta))
    retarder[2, 2] = np.cos(2 * theta)**2 * np.cos(delta) + np.sin(2 * theta)**2
    retarder[2, 3] = -np.cos(2 * theta) * np.sin(delta)

    retarder[3, 1] = -np.sin(2 * theta) * np.sin(delta)
    retarder[3, 2] = np.cos(2 * theta) * np.sin(delta)
    retarder[3, 3] = np.cos(delta)

    return retarder

def mueller_matrix_for_general_linear_polarizer(theta):
    '''Analytic expression Mueller matrix linear polarizer.

    Parameters
    ----------
    theta : scaler
        rotation angle optic in radians
    '''
    polarizer = np.zeros((4, 4))

    polarizer[0, 0] = 1
    polarizer[0, 1] = np.cos(2 * theta)
    polarizer[0, 2] = np.sin(2 * theta)

    polarizer[1, 0] = np.cos(2 * theta)
    polarizer[1, 1] = np.cos(2 * theta)**2
    polarizer[1, 2] = 0.5 * np.sin(4 * theta)

    polarizer[2, 0] = np.sin(2 * theta)
    polarizer[2, 1] = 0.5 * np.sin(4 * theta)
    polarizer[2, 2] = np.sin(2 * theta)**2

    polarizer *= 0.5
    return polarizer

def test_polarization_elements():
    N = 1
    grid = make_pupil_grid(N)
    test_field = grid.ones()

    # Stokes vectors used for testing.
    stokes_vectors = [
        None,
        np.array([1, 0, 0, 0]),  # unpolarized
        np.array([1, 1, 0, 0]),  # +Q polarized
        np.array([1, -1, 0, 0]),  # -Q polarized
        np.array([1, 0, 1, 0]),  # +U polarized
        np.array([1, 0, -1, 0]),  # -U polarized
        np.array([1, 0, 0, 1]),  # +V polarized
        np.array([1, 0, 0, -1])  # -V polarized
    ]

    # Angles of the optics that will be tested.
    angles = [-45, -22.5, 0, 22.5, 45, 90]  # degrees

    # Test QWPs, HWPs, polarizers and PBSs
    for stokes_vector in stokes_vectors:
        test_wf = Wavefront(test_field, input_stokes_vector=stokes_vector)

        if stokes_vector is None:
            # Set Stokes vector for futher calculation to unpolarized light.
            stokes_vector = np.array([1, 0, 0, 0])

        for angle in angles:
            # Create quarterwave plate.
            QWP_hcipy = QuarterWavePlate(np.radians(angle))

            # Test if Mueller matrix is the same as reference.
            QWP_ref = mueller_matrix_for_general_linear_retarder(np.radians(angle), -np.pi / 2)
            assert np.allclose(QWP_hcipy.mueller_matrix, QWP_ref)

            # Propagate wavefront through optical element.
            wf_forward_QWP = QWP_hcipy.forward(test_wf)

            # Test if result is the same as reference.
            reference_stokes_post_QWP = field_dot(QWP_ref, stokes_vector)
            assert np.allclose(wf_forward_QWP.stokes_vector[:, 0], reference_stokes_post_QWP)

            # Test backward propagation.
            wf_forward_backward_QWP = QWP_hcipy.backward(wf_forward_QWP)
            assert np.allclose(wf_forward_backward_QWP.stokes_vector[:, 0], stokes_vector)

            # Test power conservation.
            assert np.allclose(wf_forward_QWP.I, test_wf.I)

            # Create halfwave plate
            HWP_hcipy = HalfWavePlate(np.radians(angle))

            # Test if Mueller matrix is the same as reference.
            HWP_ref = mueller_matrix_for_general_linear_retarder(np.radians(angle), -np.pi)
            assert np.allclose(HWP_hcipy.mueller_matrix, HWP_ref)

            # Propagate wavefront through optical element.
            wf_forward_HWP = HWP_hcipy.forward(test_wf)

            # Test if result is the same as reference.
            reference_stokes_post_HWP = field_dot(HWP_ref, stokes_vector)
            assert np.allclose(wf_forward_HWP.stokes_vector[:, 0], reference_stokes_post_HWP)

            # Test backward propagation.
            wf_forward_backward_HWP = HWP_hcipy.backward(wf_forward_HWP)
            assert np.allclose(wf_forward_backward_HWP.stokes_vector[:, 0], stokes_vector)

            # Test power conservation.
            assert np.allclose(wf_forward_HWP.I, test_wf.I)

            # Create polarizer.
            polarizer_hcipy = LinearPolarizer(np.radians(angle))

            # Test if Mueller matrix is the same as reference.
            polarizer_ref = mueller_matrix_for_general_linear_polarizer(np.radians(angle))
            assert np.allclose(polarizer_hcipy.mueller_matrix, polarizer_ref)

            # Propagate wavefront through optical element.
            wf_forward_polarizer = polarizer_hcipy.forward(test_wf)

            # Test if result is the same as reference.
            reference_stokes_post_polarizer = field_dot(polarizer_ref, stokes_vector)
            assert np.allclose(wf_forward_polarizer.stokes_vector[:, 0], reference_stokes_post_polarizer)

            # Test backward propagation.
            wf_backward_polarizer = polarizer_hcipy.backward(test_wf)
            assert np.allclose(wf_backward_polarizer.stokes_vector[:, 0], reference_stokes_post_polarizer)

            # Create polarizing beam splitter
            LPBS_hcipy = LinearPolarizingBeamSplitter(np.radians(angle))

            # Test if Mueller matrices are the same as reference.
            polarizer_1_ref = mueller_matrix_for_general_linear_polarizer(np.radians(angle))
            polarizer_2_ref = mueller_matrix_for_general_linear_polarizer(np.radians(angle + 90))
            assert np.allclose(LPBS_hcipy.mueller_matrices[0], polarizer_1_ref)
            assert np.allclose(LPBS_hcipy.mueller_matrices[1], polarizer_2_ref)

            # Propagate wavefront through optical element.
            wf_forward_polarizer_1, wf_forward_polarizer_2 = LPBS_hcipy.forward(test_wf)

            # Test if result is the same as reference.
            reference_stokes_post_polarizer_1 = field_dot(polarizer_1_ref, stokes_vector)
            reference_stokes_post_polarizer_2 = field_dot(polarizer_2_ref, stokes_vector)
            assert np.allclose(wf_forward_polarizer_1.stokes_vector[:, 0], reference_stokes_post_polarizer_1)
            assert np.allclose(wf_forward_polarizer_2.stokes_vector[:, 0], reference_stokes_post_polarizer_2)

            # Test power conservation.
            assert np.allclose(test_wf.I, wf_forward_polarizer_1.I + wf_forward_polarizer_2.I)

            # Test multiplication of polarization optics
            # 1) JonesMatrixOpticalElement * JonesMatrixOpticalElement
            multiplication_test_1 = polarizer_hcipy * QWP_hcipy
            # 2) JonesMatrixOpticalElement * numpy array
            multiplication_test_2 = polarizer_hcipy * QWP_hcipy.jones_matrix

            multiplication_test_ref = np.dot(polarizer_ref, QWP_ref)

            # testing if the Mueller matrices are the same
            assert np.allclose(multiplication_test_1.mueller_matrix, multiplication_test_ref)
            assert np.allclose(multiplication_test_2.mueller_matrix, multiplication_test_ref)

            # propagating the wavefront through the optics
            wf_forward_multiplication_1 = multiplication_test_1.forward(test_wf)
            wf_forward_multiplication_2 = multiplication_test_2.forward(test_wf)

            reference_stokes_post_multiplication = field_dot(multiplication_test_ref, stokes_vector)

            assert np.allclose(wf_forward_multiplication_1.stokes_vector[:, 0], reference_stokes_post_multiplication)
            assert np.allclose(wf_forward_multiplication_2.stokes_vector[:, 0], reference_stokes_post_multiplication)

        # Create polarizing beam splitter
        CPBS_hcipy = CircularPolarizingBeamSplitter()

        # Test if Mueller matrices are the same as reference.
        circ_polarizer_1_ref = mueller_matrix_for_general_linear_polarizer(0)
        circ_polarizer_2_ref = mueller_matrix_for_general_linear_polarizer(np.radians(90))
        QWP_1_ref = mueller_matrix_for_general_linear_retarder(np.pi / 4, -np.pi / 2)
        CP_1_ref = np.dot(circ_polarizer_1_ref, QWP_1_ref)
        CP_2_ref = np.dot(circ_polarizer_2_ref, QWP_1_ref)

        assert np.allclose(CPBS_hcipy.mueller_matrices[0], CP_1_ref)
        assert np.allclose(CPBS_hcipy.mueller_matrices[1], CP_2_ref)

        # Propagate wavefront through optical element.
        wf_forward_circ_polarizer_1, wf_forward_circ_polarizer_2 = CPBS_hcipy.forward(test_wf)

        # Test if result is the same as reference.
        reference_stokes_post_circ_polarizer_1 = field_dot(CP_1_ref, stokes_vector)
        reference_stokes_post_circ_polarizer_2 = field_dot(CP_2_ref, stokes_vector)
        assert np.allclose(wf_forward_circ_polarizer_1.stokes_vector[:, 0], reference_stokes_post_circ_polarizer_1)
        assert np.allclose(wf_forward_circ_polarizer_2.stokes_vector[:, 0], reference_stokes_post_circ_polarizer_2)

        # Test power conservation.
        assert np.allclose(test_wf.I, wf_forward_circ_polarizer_1.I + wf_forward_circ_polarizer_2.I)

def test_magnifier():
    pupil_grid = make_pupil_grid(128)
    wf = Wavefront(make_circular_aperture(1)(pupil_grid))
    wf.total_power = 1

    magnifier = Magnifier(1.0)

    for magnification in [0.2, 3.0, [0.3, 0.3], [0.5, 2]]:
        magnifier.magnification = magnification

        wf_backward = magnifier.backward(wf)
        assert np.abs(wf_backward.total_power - 1) < 1e-12
        assert hash(wf_backward.electric_field.grid) == hash(magnifier.get_input_grid(wf.electric_field.grid, 1))

        wf_forward = magnifier.forward(wf)
        assert np.abs(wf_forward.total_power - 1) < 1e-12
        assert hash(wf_forward.electric_field.grid) == hash(magnifier.get_output_grid(wf.electric_field.grid, 1))

@pytest.mark.xfail(reason='known difficult bug; fix in progress')
def test_pickle_optical_element():
    import dill as pickle

    fname = 'optical_element.pkl'

    pupil_grid = make_pupil_grid(512)
    focal_grid = make_focal_grid(4, 16)

    elem1 = FraunhoferPropagator(pupil_grid, focal_grid)
    elem2 = SurfaceApodizer(pupil_grid.ones(), lambda wvl: 1.5 + wvl)
    elems = [elem1, elem2]

    for elem in elems:
        wf = Wavefront(pupil_grid.zeros())
        img_saved = elem(wf)

        with open(fname, 'wb') as f:
            pickle.dump(elem, f)

        with open(fname, 'rb') as f:
            elem_loaded = pickle.load(f)
            img_loaded = elem_loaded(wf)

            assert np.allclose(img_saved.electric_field, img_loaded.electric_field)

        os.remove(fname)

def test_step_index_fiber():
    core_radius_multimode = 25e-6  # m
    core_radius_singlemode = 2e-6  # m
    fiber_na = 0.13
    fiber_length = 10  # m
    D_pupil = 1  # m
    wavelength = 1e-6  # m

    multimode_fiber = StepIndexFiber(core_radius_multimode, fiber_na, fiber_length)
    singlemode_fiber = StepIndexFiber(core_radius_singlemode, fiber_na, fiber_length)

    assert np.allclose(multimode_fiber.num_modes(wavelength), 208.4953929730128)
    assert np.allclose(singlemode_fiber.num_modes(wavelength), 1.3343705150272818)

    assert np.allclose(multimode_fiber.V(wavelength), 20.42035224833366)
    assert np.allclose(singlemode_fiber.V(wavelength), 1.6336281798666927)

    assert np.allclose(multimode_fiber.mode_field_radius(wavelength), 1.6688624611223803e-05)
    assert np.allclose(singlemode_fiber.mode_field_radius(wavelength), 3.153705704872073e-06)

    core_radii = [core_radius_singlemode, core_radius_multimode]
    fibers = [singlemode_fiber, multimode_fiber]

    for fiber, core_radius in zip(fibers, core_radii):
        pupil_grid = make_pupil_grid(128)
        focal_grid = make_pupil_grid(128, 4 * core_radius)
        focal_length = D_pupil / (2 * fiber_na)

        prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

        wf = Wavefront(make_circular_aperture(D_pupil)(pupil_grid), wavelength)
        wf.total_power = 1
        img = prop(wf)

        forward_wf = fiber.forward(img)
        backward_wf = fiber.backward(img)

        assert forward_wf.total_power <= img.total_power * (1 + 1e-4)
        assert backward_wf.total_power <= forward_wf.total_power * (1 + 1e-4)

def test_gaussian_fiber_mode_power():
    for q in [16, 32, 64]:
        for mfd in [0.25, 0.5, 1]:
            grid = make_focal_grid(q, 4)
            mode = make_gaussian_fiber_mode(mfd)(grid)

            wf = Wavefront(mode)

            assert np.allclose(wf.total_power, 1)

def test_gaussian_fiber_mode_mfd():
    for mfd in [0.25, 0.5, 1]:
        grid = CartesianGrid(UnstructuredCoords(([0, mfd / 2], [0, 0])))
        mode = make_gaussian_fiber_mode(mfd)(grid)

        assert np.allclose(mode[1] / mode[0], np.exp(-1))

def test_single_mode_fiber_injection():
    grid = make_focal_grid(16, 4)
    mode_1 = make_gaussian_fiber_mode(1)

    def mode_2(grid):
        return mode_1(grid) * 1j

    for mode in [mode_1, mode_2]:
        smf = SingleModeFiberInjection(grid, mode)

        wf = Wavefront(mode(grid))
        wf.total_power = 1

        post_fiber_wf = smf.forward(wf)

        # This wavefront should fully couple into the fiber.
        assert np.allclose(post_fiber_wf.electric_field, 1)
        assert np.allclose(post_fiber_wf.total_power, 1)

        pre_fiber_wf = smf.backward(post_fiber_wf)

        # Power should be conserved when injecting a perfect mode.
        assert np.allclose(pre_fiber_wf.electric_field, wf.electric_field)
        assert np.allclose(pre_fiber_wf.total_power, 1)

        wf = Wavefront(mode(grid) * grid.x)
        wf.total_power = 1

        post_fiber_wf = smf.forward(wf)

        # This wavefront should not couple at all into the fiber.
        assert np.allclose(post_fiber_wf.total_power, 0)

def check_thin_lens(wf, lens, num_steps, focal_length):
    dz = focal_length / (num_steps / 2)
    prop = FresnelPropagator(wf.electric_field.grid, dz)

    wavefront = lens(wf)

    z = [0]
    peak = [wavefront.power.max()]

    for i in range(num_steps):
        wavefront = prop(wavefront)

        z.append(z[-1] + dz)
        peak.append(wavefront.power.max())

    print('z', z)
    print('peak', peak)

    return z, peak

def test_thin_lens():
    wavelength = 1e-6
    pupil_diameter = 5e-2
    num_steps = 20

    grid = make_pupil_grid(256, 2 * pupil_diameter)

    focal_length = 300e-1
    lens = ThinLens(focal_length, lambda x: 1.5, 1e-6)

    aperture = evaluate_supersampled(make_circular_aperture(pupil_diameter), grid, 8)
    assert abs((lens.focal_length - focal_length) / focal_length) < 1e-10

    wf = Wavefront(aperture, wavelength)
    wf.total_power = 1

    z1, peak1 = check_thin_lens(wf, lens, num_steps, focal_length)

    lens.focal_length = 2 * focal_length
    z2, peak2 = check_thin_lens(wf, lens, num_steps, 2 * focal_length)

    lens.focal_length = 0.75 * focal_length
    z3, peak3 = check_thin_lens(wf, lens, num_steps, 0.75 * focal_length)

    assert abs(z1[np.argmax(peak1)] / focal_length - 1.0) < 0.01
    assert abs(z2[np.argmax(peak2)] / (2 * focal_length) - 1.0) < 0.01
    assert abs(z3[np.argmax(peak3)] / (0.75 * focal_length) - 1.0) < 0.01

def test_phase_grating():
    from scipy.special import jv
    grid = make_pupil_grid(128, 1)
    aperture = make_circular_aperture(1)(grid)

    cor = PerfectCoronagraph(aperture)
    wf = Wavefront(aperture)
    wf.total_power = 1.0

    amplitudes = np.linspace(0, np.pi, 31)
    diffraction_efficiency = np.array([cor(PhaseGrating(1 / 20, amp)(wf)).total_power for amp in amplitudes])
    eta = 1 - jv(0, amplitudes)**2
    err = eta - diffraction_efficiency

    assert np.allclose(err, 0, atol=1e-3)

    grating = PhaseGrating(1 / 20, np.pi / 2, lambda grid: np.sign(np.sin(2 * np.pi * grid.y)), orientation=0)
    wf_grating = cor(grating(wf))
    assert abs(wf_grating.total_power - 1) < 1e-15

    grating.orientation = np.pi / 2
    wf_grating = cor(grating(wf))
    assert abs(wf_grating.total_power - 1) < 1e-15

def test_thin_prism():
    nbk7 = get_refractive_index("N-BK7")
    wedge_angle = np.deg2rad(3. + 53.0 / 60.0)
    thin_prism = ThinPrism(wedge_angle, nbk7)

    test_wavelengths = np.linspace(500, 1000, 31) * 1e-9
    err = np.array([thin_prism.trace(wave) - (nbk7(wave) - 1) * wedge_angle for wave in test_wavelengths])

    assert np.allclose(err, 0, atol=1e-3)

def test_prism():
    nbk7 = get_refractive_index("N-BK7")
    wedge_angle = np.deg2rad(3. + 53.0 / 60.0)
    prism = Prism(0, wedge_angle, nbk7)

    test_wavelengths = np.array([650.0, 700.0, 750.0]) * 1e-9
    deviation_angles = np.deg2rad(np.array([2.00390481945681, 1.9982250862238, 1.9934348068561]))
    err = np.array([prism.trace(wave) - dev for wave, dev in zip(test_wavelengths, deviation_angles)])

    assert np.allclose(np.rad2deg(err), 0, atol=1e-3)
    assert np.allclose(prism.minimal_deviation_angle(750e-9), 0.03471595671032256)

    prism.prism_angle = 2 * np.deg2rad(3. + 53.0 / 60.0)
    assert np.allclose(prism.minimal_deviation_angle(750e-9), 0.06958405951915544)

    Dtel = 1
    Dgrid = 1.1 * Dtel
    sr = np.mean(test_wavelengths) / Dtel

    grid = make_pupil_grid(128, Dgrid)
    aperture = make_circular_aperture(Dtel)(grid)

    focal_grid = make_focal_grid(q=3, num_airy=15, spatial_resolution=sr)
    prop = FraunhoferPropagator(grid, focal_grid)

    for wi, wave in enumerate(test_wavelengths):
        wf = Wavefront(aperture, wave)
        wf.total_power = 1

        prism.orientation = 2 * np.pi / 3 * wi
        prism_compensation = TiltElement(-prism.trace(wave), orientation=2 * np.pi / 3 * wi)
        wf_deviated = prop(prism_compensation(prism(wf)))
        wf_foc = prop(wf)

        assert np.max(abs(wf_deviated.power - wf_foc.power) / wf_foc.power.max()) < 1e-8
