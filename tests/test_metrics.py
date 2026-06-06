import numpy as np
import pytest

import hcipy

@pytest.fixture
def gaussian_image():
    q = 32.0
    span = 8.0
    g = hcipy.make_focal_grid(np.float64(q), np.float64(span))
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    sigma = 1.0
    img = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    sep_x = g.separated_coords[0]
    sep_y = g.separated_coords[1]
    delta_x = float(sep_x[1] - sep_x[0])
    delta_y = float(sep_y[1] - sep_y[0])
    return hcipy.Field(img, g), delta_x, delta_y, sigma

def test_sub_pixel_peak_centered(gaussian_image):
    field, delta_x, delta_y, _ = gaussian_image
    x_peak, y_peak = hcipy.sub_pixel_peak(field)
    assert abs(x_peak) < 0.05
    assert abs(y_peak) < 0.05

def test_sub_pixel_peak_shifted(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    sub_dx, sub_dy = 0.5 * delta_x, -0.3 * delta_y
    img = np.exp(-((xs - sub_dx) ** 2 + (ys - sub_dy) ** 2) / (2 * sigma ** 2))
    field = hcipy.Field(img, g)
    x_peak, y_peak = hcipy.sub_pixel_peak(field)
    assert abs(x_peak - sub_dx) < 0.05
    assert abs(y_peak - sub_dy) < 0.05

def test_sub_pixel_peak_with_mask(gaussian_image):
    field, delta_x, delta_y, _ = gaussian_image
    g = field.grid
    Ny, Nx = g.shape
    mask_field = hcipy.Field(np.ones((Ny, Nx), dtype=bool), g)
    x_peak, y_peak = hcipy.sub_pixel_peak(field, mask=mask_field)
    assert abs(x_peak) < 0.05
    assert abs(y_peak) < 0.05

def test_sub_pixel_peak_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.sub_pixel_peak(field + 1j * field)

def test_centroid_centered(gaussian_image):
    field, _, _, _ = gaussian_image
    x_c, y_c = hcipy.centroid(field)
    assert abs(x_c) < 1e-6
    assert abs(y_c) < 1e-6

def test_centroid_shifted(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    img = np.exp(-((xs - 0.7 * delta_x) ** 2 + (ys + 0.2 * delta_y) ** 2) / (2 * sigma ** 2))
    field = hcipy.Field(img, g)
    x_c, y_c = hcipy.centroid(field)
    assert abs(x_c - 0.7 * delta_x) < 0.01 * delta_x
    assert abs(y_c + 0.2 * delta_y) < 0.01 * delta_y

def test_centroid_with_mask(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    Ny, Nx = g.shape
    r = np.sqrt((xs - 0) ** 2 + (ys - 0) ** 2).reshape(Ny, Nx)
    mask = hcipy.Field(r < 2.0, g)
    x_c, y_c = hcipy.centroid(field, mask=mask)
    assert abs(x_c) < 1e-6
    assert abs(y_c) < 1e-6

def test_centroid_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.centroid(field + 1j * field)

def test_fwhm_gaussian(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    fwhm_value = hcipy.fwhm(field)
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert abs(fwhm_value - expected) < 0.01 * expected

def test_fwhm_unknown_method(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(ValueError):
        hcipy.fwhm(field, method='unknown')

def test_fwhm_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.fwhm(field + 1j * field)

def test_ellipticity_circular(gaussian_image):
    field, _, _, _ = gaussian_image
    a, b, theta, ell = hcipy.ellipticity(field)
    assert abs(a - b) < 1e-3 * a
    assert abs(ell) < 1e-6

def test_ellipticity_stretched(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    stretch = 2.0
    img = np.exp(-((xs / stretch) ** 2 + ys ** 2) / (2 * sigma ** 2))
    field = hcipy.Field(img, g)
    a, b, theta, ell = hcipy.ellipticity(field)
    assert abs(a / b - stretch) < 0.05 * stretch
    assert abs(ell - (1 - 1 / stretch)) < 0.01

def test_ellipticity_with_mask(gaussian_image):
    field, _, _, _ = gaussian_image
    g = field.grid
    Ny, Nx = g.shape
    mask = hcipy.Field(np.ones((Ny, Nx), dtype=bool), g)
    a, b, theta, ell = hcipy.ellipticity(field, mask=mask)
    assert abs(a - b) < 1e-3 * a

def test_ellipticity_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.ellipticity(field + 1j * field)

def test_image_shift_zero(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    dx, dy = hcipy.image_shift(field, field)
    assert abs(dx) < 1e-10
    assert abs(dy) < 1e-10

def test_image_shift_integer(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    ref_arr = np.exp(-((xs - 2 * delta_x) ** 2 + (ys + 1 * delta_y) ** 2) / (2 * sigma ** 2))
    ref = hcipy.Field(ref_arr, g)
    dx, dy = hcipy.image_shift(field, ref)
    assert abs(dx - (-2 * delta_x)) < 1e-10
    assert abs(dy - (1 * delta_y)) < 1e-10

def test_image_shift_subpixel(gaussian_image):
    field, delta_x, delta_y, sigma = gaussian_image
    g = field.grid
    xs = np.asarray(g.x.data)
    ys = np.asarray(g.y.data)
    sub_dx, sub_dy = 0.5 * delta_x, -0.3 * delta_y
    ref_arr = np.exp(-((xs - sub_dx) ** 2 + (ys - sub_dy) ** 2) / (2 * sigma ** 2))
    ref = hcipy.Field(ref_arr, g)
    dx, dy = hcipy.image_shift(field, ref)
    err = np.sqrt((dx - (-sub_dx)) ** 2 + (dy - (-sub_dy)) ** 2)
    assert err < 0.1 * max(delta_x, delta_y)

def test_image_shift_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.image_shift(field + 1j * field, field)

def test_encircled_energy_gaussian(gaussian_image):
    field, _, _, sigma = gaussian_image
    radii, ee = hcipy.encircled_energy(field)
    assert radii.shape[0] == ee.shape[0]
    assert float(radii[0]) == 0
    assert np.all(np.diff(radii.data) >= 0)
    assert float(ee[0]) >= 0
    assert abs(float(ee[-1]) - 1.0) < 1e-6
    idx = int(np.argmin(np.abs(np.asarray(radii.data) - sigma)))
    assert abs(float(ee[idx]) - (1 - np.exp(-0.5))) < 0.01

def test_encircled_energy_with_mask(gaussian_image):
    field, _, _, sigma = gaussian_image
    g = field.grid
    Ny, Nx = g.shape
    mask = hcipy.Field(np.ones((Ny, Nx), dtype=bool), g)
    radii, ee = hcipy.encircled_energy(field, mask=mask)
    assert abs(float(ee[-1]) - 1.0) < 1e-6

def test_encircled_energy_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.encircled_energy(field + 1j * field)

def test_ensquared_energy_gaussian(gaussian_image):
    field, _, _, _ = gaussian_image
    halfwidths, ee = hcipy.ensquared_energy(field)
    assert halfwidths.shape[0] == ee.shape[0]
    assert float(halfwidths[0]) == 0
    assert float(ee[0]) >= 0
    assert abs(float(ee[-1]) - 1.0) < 1e-6
    assert np.all(np.diff(ee.data) >= -1e-10)

def test_ensquared_energy_rejects_complex(gaussian_image):
    field, _, _, _ = gaussian_image
    with pytest.raises(TypeError):
        hcipy.ensquared_energy(field + 1j * field)
