import numpy as np
from hcipy import *
import scipy.special
import pytest

def test_fraunhofer_propagation_circular():
    for num_pix in [512, 1024]:
        pupil_grid = make_pupil_grid(num_pix)
        focal_grid = make_focal_grid(16, 8)

        ind = focal_grid.closest_to((0, 0))

        for diameter in [1, 0.7]:
            aperture = evaluate_supersampled(make_circular_aperture(diameter), pupil_grid, 8)

            for focal_length in [1, 0.8]:
                prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

                for wavelength in [1, 0.5]:
                    wf = Wavefront(aperture, wavelength)
                    img = prop(wf).electric_field
                    img /= img[np.argmax(np.abs(img))]

                    x = focal_grid.as_('polar').r * np.pi / wavelength * diameter / focal_length
                    x[ind] = 1
                    reference = 2 * scipy.special.jv(1, x) / x
                    reference[ind] = 1

                    if num_pix == 512:
                        assert np.abs(img - reference).max() < 3e-5
                    elif num_pix == 1024:
                        assert np.abs(img - reference).max() < 1e-5
                    else:
                        # This should never happen.
                        assert False

def test_fraunhofer_propagation_rectangular():
    for num_pix in [512, 1024]:
        pupil_grid = make_pupil_grid(num_pix)
        focal_grid = make_focal_grid(16, 8)

        for size in [[1, 1], [0.75, 1], [0.75, 0.75]]:
            aperture = evaluate_supersampled(make_rectangular_aperture(size), pupil_grid, 8)

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
                        # This should never happen.
                        assert False

@pytest.mark.parametrize('propagator', [FresnelPropagator, AngularSpectrumPropagator])
@pytest.mark.parametrize('number_of_pixels', [
    pytest.param(512),
    pytest.param(1024, marks=pytest.mark.slow)
])
@pytest.mark.parametrize('wavelength', [
    pytest.param(500e-9),
    pytest.param(700e-9, marks=pytest.mark.slow)
])
@pytest.mark.parametrize('a, b', [
    pytest.param(0.001, 0.001),
    pytest.param(0.0015, 0.001, marks=pytest.mark.slow)
])
@pytest.mark.parametrize('relative_distance,error_threshold', [
    pytest.param(0.2, 0.02),
    pytest.param(1, 0.01, marks=pytest.mark.slow),
    pytest.param(5, 0.01)
])
def test_nearfield_propagation_rectangular(propagator, number_of_pixels, wavelength, a, b, relative_distance, error_threshold):
    wavenumber = 2 * np.pi / wavelength

    pupil_grid = make_pupil_grid(number_of_pixels, [16 * a, 16 * b])
    threshold_distance = np.min(pupil_grid.delta * np.array([16 * a, 16 * b]) / wavelength)
    distance = relative_distance * threshold_distance

    prop_pos = propagator(pupil_grid, distance, num_oversampling=2)
    prop_neg = propagator(pupil_grid, -distance, num_oversampling=2)

    aperture = evaluate_supersampled(make_rectangular_aperture([2 * a, 2 * b]), pupil_grid, 2)

    img_forward = prop_pos.forward(Wavefront(aperture, wavelength)).intensity
    img_backward = prop_pos.backward(Wavefront(aperture, wavelength)).intensity

    img_forward_neg = prop_neg.forward(Wavefront(aperture, wavelength)).intensity
    img_backward_neg = prop_neg.forward(Wavefront(aperture, wavelength)).intensity

    assert np.allclose(img_forward, img_backward_neg)
    assert np.allclose(img_backward, img_forward_neg)

    def generate_reference_field(reference_grid):
        w_x1 = np.sqrt(2 / (distance * wavelength)) * (reference_grid.x - a)
        w_x2 = np.sqrt(2 / (distance * wavelength)) * (reference_grid.x + a)
        w_y1 = np.sqrt(2 / (distance * wavelength)) * (reference_grid.y - b)
        w_y2 = np.sqrt(2 / (distance * wavelength)) * (reference_grid.y + b)

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
        reference *= -1j * np.exp(1j * wavenumber * distance) / 2
        reference *= np.exp(1j * wavenumber * (reference_grid.x**2 + reference_grid.y**2) / (2 * distance))
        reference = np.abs(Field(reference, reference_grid))**2

        return reference

    reference_image = evaluate_supersampled(generate_reference_field, pupil_grid, 2)

    img_forward = subsample_field(img_forward, 8)
    img_backward = subsample_field(img_backward, 8)
    reference_image = subsample_field(reference_image, 8)

    absolute_error_forward = np.abs(img_forward - reference_image).max()
    assert absolute_error_forward < error_threshold

    absolute_error_backward = np.abs(img_backward - reference_image).max()
    assert absolute_error_backward < error_threshold
