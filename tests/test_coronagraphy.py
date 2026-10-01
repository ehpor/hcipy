from hcipy import *
import numpy as np

def test_vortex_coronagraph():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    aperture = make_circular_aperture(1)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    lyot = make_circular_aperture(0.99)
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

def test_fqpm_coronagraph():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    aperture = make_circular_aperture(1)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    lyot = make_circular_aperture(0.99)
    lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

    fqpm = FQPMCoronagraph(pupil_grid)

    wf = Wavefront(aperture)
    wf.total_power = 1

    img_ref = prop(wf)

    wf = fqpm(wf)
    wf.electric_field *= lyot
    img = prop(wf)

    assert img.total_power < 1e-6
    assert img.intensity.max() / img_ref.intensity.max() < 4e-8


def test_lyot_coronagraph_with_phase_apodizer_mask():
    pupil_grid = make_pupil_grid(32)
    focal_grid = make_focal_grid(4, 16)

    phase = make_circular_aperture(1.0)(focal_grid) * 0.3
    mask = PhaseApodizer(phase)

    coro = LyotCoronagraph(pupil_grid, mask)

    assert coro.prop.get_instance_data(pupil_grid, None, 1).output_grid is focal_grid


def test_lyot_coronagraph_with_polarization_mask():
    pupil_grid = make_pupil_grid(32)
    focal_grid = make_focal_grid(4, 16)

    phase_retardation = make_circular_aperture(1.0)(focal_grid) * 0.3
    fast_axis_orientation = Field(np.zeros(focal_grid.size), focal_grid)
    mask = LinearRetarder(phase_retardation, fast_axis_orientation)

    coro = LyotCoronagraph(pupil_grid, mask)

    assert coro.prop.get_instance_data(pupil_grid, None, 1).output_grid is focal_grid


def test_vector_vortex_coronagraph():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    aperture = make_circular_aperture(1)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    lyot = make_circular_aperture(0.99)
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

        # Backward check
        # with input 2-tensor electric field
        img.electric_field[0, 1] *= 0
        img.electric_field[1, 0] *= 0
        img.electric_field[1, 1] *= 0
        wf_b = prop.backward(img)
        wf_input_b = vortex.backward(wf_b)

        # with input scalar electric field
        wf_c = prop.backward(Wavefront(img.electric_field[0, 0]))
        wf_input_c = vortex.backward(wf_c)
        assert np.allclose(wf_input_b.electric_field[0, 0], wf_input_c.electric_field[0, 0])

def test_vector_vortex_coronagraph_polarized():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    aperture = make_obstructed_circular_aperture(1, 0.2)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    lyot = make_obstructed_circular_aperture(0.99, 0.2 / 0.99)
    lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

    stokes_vector = (1, 0, 0, 1)
    for charge in [2]:
        vortex = VectorVortexCoronagraph(charge)

        wf = Wavefront(aperture, input_stokes_vector=stokes_vector)
        wf.total_power = 1

        wf = vortex(wf)
        wf.electric_field *= lyot
        img = prop(wf)

        # incident circular polarization is transfered to orthogonal state
        assert np.allclose(img.intensity, (-1) * img.V)

        # Backward check
        wf_b = prop.backward(img)
        wf_input_b = vortex.backward(wf_b)
        assert np.allclose(wf_input_b.intensity, (-1)**2 * wf_input_b.V)

def test_scalar_vortex_coronagraph_polarized():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    aperture = make_obstructed_circular_aperture(1, 0.2)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    lyot = make_obstructed_circular_aperture(0.99, 0.2 / 0.99)
    lyot = evaluate_supersampled(lyot, pupil_grid, 8) > 1 - 1e-5

    stokes_vector = (1, 0, 0, 1)
    for charge in [2]:
        vortex = VortexCoronagraph(pupil_grid, charge)

        wf = Wavefront(aperture, input_stokes_vector=stokes_vector)
        wf.total_power = 1

        wf = vortex(wf)
        wf.electric_field *= lyot
        img = prop(wf)

        # polarization state preserved
        assert np.allclose(img.intensity, img.V)

        wf_b = prop.backward(img)
        wf_input_b = vortex.backward(wf_b)
        assert np.allclose(wf_input_b.intensity, wf_input_b.V)

def test_ravc():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 32)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    for co in [0.1, 0.2, 0.3]:
        aperture = lambda grid: make_circular_aperture(1)(grid) - make_circular_aperture(co)(grid)
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

    aperture = evaluate_supersampled(make_circular_aperture(1), pupil_grid, 8)
    wavefront = Wavefront(aperture)
    wavefront.total_power = 1

    # reference PSF without APP
    img_ref = propagator.forward(wavefront)

    # small rectangular dark zone with 1e-7 contrast
    mask = make_rectangular_aperture(size=(6, 2), center=(9, 0))(focal_grid)
    contrast = 1 - mask + 1e-7

    # APP with aggressive acceleration
    app = generate_app_keller(wavefront, propagator, contrast, num_iterations=150, beta=0.98)
    img = propagator.forward(app)

    assert np.abs(img.intensity.max() / img_ref.intensity.max() - 0.947) < 0.01  # Strehl
    assert np.mean(img.intensity * mask) / np.mean(mask) < 1.6e-8  # contrast

def test_perfect_coronagraph():
    pupil_grid = make_pupil_grid(256)
    aperture = make_circular_aperture(1)(pupil_grid)

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

def _reference_coronagraph_operator(aperture, order, coeffs):
    '''Compute the explicit (non-separable) projection operator for reference.'''
    pupil_grid = aperture.grid
    p = order // 2

    modes = [aperture * pupil_grid.x**j * pupil_grid.y**(i - j) for i in range(p) for j in range(i + 1)]
    mode_matrix = np.stack([np.asarray(mode) for mode in modes], axis=-1)

    q, r = np.linalg.qr(mode_matrix)
    r = r * np.sign(np.diag(r))[:, np.newaxis]
    r_inverse = np.linalg.inv(r)

    k = r_inverse.dot(np.diag(coeffs)).dot(r_inverse.conj().T)

    return mode_matrix, k

def _check_coronagraph_against_reference(aperture, order, coeffs, field_shape, check_transformation=True):
    rng = np.random.default_rng(0)
    pupil_grid = aperture.grid

    coro = PerfectCoronagraph(aperture, order, coeffs=coeffs)
    mode_matrix, k = _reference_coronagraph_operator(aperture, order, coro.coeffs)

    electric_field = rng.standard_normal(field_shape + (pupil_grid.size,)) + 1j * rng.standard_normal(field_shape + (pupil_grid.size,))
    flattened = electric_field.reshape(-1, pupil_grid.size)
    correction = (mode_matrix.dot(k.dot(mode_matrix.conj().T.dot(flattened.T)))).T.reshape(electric_field.shape)

    if len(field_shape) == 2:
        wavefront = Wavefront(Field(electric_field, pupil_grid), input_stokes_vector=[1, 0, 0, 0])
    else:
        wavefront = Wavefront(Field(electric_field, pupil_grid))

    result = coro(wavefront).electric_field

    assert np.allclose(electric_field - np.asarray(result), correction, rtol=1e-8, atol=1e-10)

    if check_transformation:
        expected_transformation = np.eye(pupil_grid.size) - mode_matrix.dot(k.dot(mode_matrix.conj().T))
        assert np.allclose(coro.get_transformation_matrix_forward(), expected_transformation, rtol=1e-8, atol=1e-10)

def test_perfect_coronagraph_separable():
    rng = np.random.default_rng(0)
    pupil_grid = make_pupil_grid((48, 64))

    # An asymmetric aperture catches any mixing up of the x- and y-axes.
    aperture = make_circular_aperture(0.9)(pupil_grid) * (1 + 0.3 * np.tanh(pupil_grid.x)) * np.exp(0.2j * pupil_grid.y)

    # Order 2 deliberately uses the explicit path, so only the separable orders are tested here.
    for order in [4, 6, 8]:
        num_modes = int(order * (order / 2 + 1) / 4)
        for coeffs in [None, rng.random(num_modes)]:
            for field_shape in [(), (2,), (2, 2)]:
                check_transformation = field_shape == () and coeffs is None
                _check_coronagraph_against_reference(aperture, order, coeffs, field_shape, check_transformation)

def test_perfect_coronagraph_separated_coords_grid():
    # Separated grids also have shape (ny, nx); both non-square and square grids must work.
    for nx, ny in [(48, 64), (64, 48), (64, 64)]:
        pupil_grid = CartesianGrid(SeparatedCoords([np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)]))
        aperture = make_circular_aperture(0.9)(pupil_grid)

        for order in [2, 4, 6]:
            num_modes = int(order * (order / 2 + 1) / 4)
            _check_coronagraph_against_reference(aperture, order, None, (), check_transformation=(order == 2))
            _check_coronagraph_against_reference(aperture, order, np.linspace(0.5, 1, num_modes), (2,), check_transformation=False)

def test_perfect_coronagraph_non_separated_grid():
    pupil_grid = make_pupil_grid(64)
    aperture = make_circular_aperture(0.9)(pupil_grid)
    subset_grid = pupil_grid.subset(np.asarray(aperture) > 0.5)
    subset_aperture = make_circular_aperture(0.9)(subset_grid)

    assert not subset_grid.is_separated

    for order in [2, 4, 6]:
        _check_coronagraph_against_reference(subset_aperture, order, None, (), check_transformation=(order == 2))

def test_lyot_coronagraph():
    pupil_grid = make_pupil_grid(128, 1.1)
    aperture = evaluate_supersampled(make_circular_aperture(1.0), pupil_grid, 8)
    lyot_stop = evaluate_supersampled(make_circular_aperture(0.95), pupil_grid, 8)

    # Coronagraph 1 with the default internal focal length
    fpm_grid = make_focal_grid(q=32, num_airy=3)
    fpm = 1 - evaluate_supersampled(make_circular_aperture(5), fpm_grid, 8)
    cor = LyotCoronagraph(pupil_grid, fpm, lyot_stop)

    # Coronagraph 2 with a large focal length
    focal_length = 10.0
    fpm_grid2 = make_focal_grid(q=32, num_airy=3, spatial_resolution=focal_length)
    fpm2 = 1 - evaluate_supersampled(make_circular_aperture(5 * focal_length), fpm_grid2, 8)
    cor2 = LyotCoronagraph(pupil_grid, fpm2, lyot_stop, focal_length=focal_length)

    # Coronagraph 3 with optical element for focal plane mask
    cor3 = LyotCoronagraph(pupil_grid, Apodizer(fpm), lyot_stop)

    # The grid on which the performance is evaluated
    focal_grid = make_focal_grid(q=3, num_airy=25)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    wf = Wavefront(aperture)
    wf.total_power = 1
    norm = prop(wf).power.max()
    wf_foc = prop(cor(wf))
    wf_foc2 = prop(cor2(wf))
    wf_foc3 = prop(cor3(wf))
    # Checks performance of the coronagraph and if the focal length does not introduce artifacts
    assert (wf_foc.power.max() / norm) < 5e-3
    np.testing.assert_allclose(wf_foc.power, wf_foc2.power)
    np.testing.assert_allclose(wf_foc.power, wf_foc3.power)

def test_vortex_fiber_nuller():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 5)

    aperture = make_circular_aperture(1)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    fiber = SingleModeFiberInjection(focal_grid, make_gaussian_fiber_mode(1.4))

    for charge in [1, 2]:
        vfn = VortexFiberNuller(pupil_grid, fiber, charge)
        wf = Wavefront(aperture)
        wf.total_power = 1

        img = vfn(wf)

        assert img.total_power < 1e-8

def test_photonic_lantern_nuller():
    pupil_grid = make_pupil_grid(256)
    focal_grid = make_focal_grid(4, 5)

    aperture = make_circular_aperture(1)
    aperture = evaluate_supersampled(aperture, pupil_grid, 8)

    pln = PhotonicLanternNuller(pupil_grid, focal_grid)

    wf = Wavefront(aperture)
    wf.total_power = 1

    img = pln(wf)

    assert np.allclose(img.intensity[1:5], np.zeros(4))

def test_knife_edge_coronagraph():
    grid = make_pupil_grid(64, 1.1)
    aperture = make_circular_aperture(1)(grid)

    focal_grid = make_focal_grid(q=5, num_airy=5)

    prop = FraunhoferPropagator(grid, focal_grid)
    lyot_aperture = make_circular_aperture(0.95)(grid)
    lyot_stop = Apodizer(lyot_aperture)

    wf = Wavefront(aperture)
    wf.total_power = 1.0
    norm = prop(lyot_stop(wf)).power.max()

    directions = ['+x', '-x', '+y', '-y']
    for direction in directions:
        knife_edge = KnifeEdgeLyotCoronagraph(grid, direction=direction, apodizer=None, lyot_stop=lyot_aperture)
        wf_cor = prop(knife_edge(wf))

        assert (wf_cor.power.max() / norm) < 0.3

    directions = ['+x', '-x', '+y', '-y']
    knife_edge_shifts = [-1.0 * grid.x, 1.0 * grid.x, -1.0 * grid.y, 1.0 * grid.y]
    for shift, direction in zip(knife_edge_shifts, directions):
        pre_apodizer = np.exp(1j * 2 * np.pi * shift)
        knife_edge = KnifeEdgeLyotCoronagraph(grid, direction=direction, apodizer=pre_apodizer, lyot_stop=lyot_aperture * np.conj(pre_apodizer))
        wf_cor = prop(knife_edge(wf))

        assert (wf_cor.power.max() / norm) < 1e-2

    # Test for symmetry between a left and right knige edge
    knife_edge_left = KnifeEdgeLyotCoronagraph(grid, direction='+x', lyot_stop=lyot_aperture)
    wf_left = prop(knife_edge_left(wf))

    knife_edge_right = KnifeEdgeLyotCoronagraph(grid, direction='-x', lyot_stop=lyot_aperture)
    wf_right = prop(knife_edge_right(wf))

    # The PSF needs to be flipped and shifted by 1 pixel because the focal grid is odd
    flipped_psf = np.roll(wf_right.power.shaped[:, ::-1], 1, axis=1)
    flipped_psf = Field(flipped_psf.ravel(), focal_grid)

    # Ignore the first column in the evaluation because of the roll over effect.
    assert (abs(wf_left.power.shaped[:, 1::] - flipped_psf.shaped[:, 1::]).max() / norm) < 1e-12

    # Test for symmetry between an up and down knige edge
    knife_edge_down = KnifeEdgeLyotCoronagraph(grid, direction='+y', lyot_stop=lyot_aperture)
    wf_down = prop(knife_edge_down(wf))

    knife_edge_up = KnifeEdgeLyotCoronagraph(grid, direction='-y', lyot_stop=lyot_aperture)
    wf_up = prop(knife_edge_up(wf))

    # The PSF needs to be flipped and shifted by 1 pixel because the focal grid is odd
    flipped_psf = np.roll(wf_up.power.shaped[::-1, :], 1, axis=0)
    flipped_psf = Field(flipped_psf.ravel(), focal_grid)

    # Ignore the first row in the evaluation because of the roll over effect.
    assert (abs(wf_down.power.shaped[1::, :] - flipped_psf.shaped[1::, :]).max() / norm) < 1e-12
