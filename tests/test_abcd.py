import pytest
import array_api_compat
from hcipy import *
import array_api_strict as xp
from hcipy._math.backends import all_close
import math

WL = 500e-9

def test_make_abcd_free_space():
    z = 0.5
    M = make_abcd_free_space(z, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[1, z], [0, 1]]))

def test_make_abcd_thin_lens():
    f = 2.0
    M = make_abcd_thin_lens(f, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[1, 0], [-1 / f, 1]]))

def test_make_abcd_mirror():
    radius = 4.0
    M = make_abcd_mirror(radius, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[1, 0], [-2 / radius, 1]]))

    # A plane mirror (radius=None) is the identity.
    M = make_abcd_mirror(xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.eye(2))

def test_make_abcd_fraunhofer():
    f = 2.0
    M = make_abcd_fraunhofer(f, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[0, f], [-1 / f, 0]]))

def test_make_abcd_magnifier():
    magnification = 3.0
    M = make_abcd_magnifier(magnification, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[magnification, 0], [0, 1 / magnification]]))

def test_make_abcd_refractive_interface():
    n1, n2, radius = 1.0, 1.5, 2.0
    M = make_abcd_refractive_interface(n1, n2, radius, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[1, 0], [(n1 - n2) / (radius * n2), n1 / n2]]))

    # A flat interface is obtained with radius=None (the default).
    M = make_abcd_refractive_interface(n1, n2, xp=xp)

    assert isinstance(M, RayTransferMatrix)
    assert all_close(M(WL), xp.asarray([[1, 0], [0, n1 / n2]]))

def test_ray_transfer_matrix_composition():
    f, z = 2.0, 0.5

    M_sys = make_abcd_free_space(z, xp=xp) @ make_abcd_thin_lens(f, xp=xp)

    expected = make_abcd_free_space(z, xp=xp)(WL) @ make_abcd_thin_lens(f, xp=xp)(WL)
    assert all_close(M_sys(WL), expected)
    assert not M_sys.is_dispersive
    assert array_api_compat.array_namespace(M_sys(WL)) is xp

    # Composition with a raw array works as well.
    M = make_abcd_thin_lens(f, xp=xp) @ xp.eye(2)
    assert all_close(M(WL), make_abcd_thin_lens(f, xp=xp)(WL))

def test_ray_transfer_matrix_predicates():
    assert make_abcd_thin_lens(1.0, xp=xp).is_surface(WL)
    assert not make_abcd_free_space(1.0, xp=xp).is_surface(WL)

    assert make_abcd_free_space(0.5, xp=xp).is_free_space(WL)
    assert not make_abcd_thin_lens(1.0, xp=xp).is_free_space(WL)

def test_ray_transfer_matrix_derived_scalars():
    f = 2.0
    M = make_abcd_thin_lens(f, xp=xp)

    assert float(M.effective_focal_length(WL)) == pytest.approx(f)
    assert float(M.paraxial_power(WL)) == pytest.approx(1 / f)
    assert float(M.geometric_magnification(WL)) == pytest.approx(1.0)
    assert float(M.angular_magnification(WL)) == pytest.approx(1.0)
    assert float(M.distance(WL)) == pytest.approx(0.0)
    assert float(M.front_focal_length(WL)) == pytest.approx(f)
    assert float(M.back_focal_length(WL)) == pytest.approx(f)

    M = make_abcd_free_space(0.5, xp=xp)
    assert float(M.distance(WL)) == pytest.approx(0.5)

    M = make_abcd_magnifier(2.0, xp=xp)
    assert float(M.geometric_magnification(WL)) == pytest.approx(2.0)
    assert float(M.angular_magnification(WL)) == pytest.approx(0.5)

def test_ray_transfer_matrix_requires_wavelength():
    M = make_abcd_free_space(0.5, xp=xp)

    with pytest.raises(TypeError):
        M()
    with pytest.raises(TypeError):
        M.effective_focal_length()
    with pytest.raises(TypeError):
        M.is_surface()

    M = make_abcd_refractive_interface(lambda wl: 1.5, 1.0, xp=xp)
    with pytest.raises(TypeError):
        M()

def test_ray_transfer_matrix_shape_validation():
    with pytest.raises(ValueError):
        RayTransferMatrix(xp.eye(3))

def test_ray_transfer_matrix_wraps_array_and_callable():
    M = RayTransferMatrix(xp.asarray([[1.0, 0.5], [0.0, 1.0]]))
    assert not M.is_dispersive
    assert all_close(M(WL), xp.asarray([[1, 0.5], [0, 1]]))

    M = RayTransferMatrix(lambda wl: xp.asarray([[1.0, wl], [0.0, 1.0]]))
    assert M.is_dispersive
    assert all_close(M(WL), xp.asarray([[1, WL], [0, 1]]))

def test_dispersive_ray_transfer_matrix():
    n = make_sellmeier_glass(1, [0.6961663, 0.4079426, 0.8974794], [0.0684043**2, 0.1162414**2, 9.896161**2])

    M = make_abcd_refractive_interface(n, 1.0, xp=xp)
    assert isinstance(M, RayTransferMatrix)
    assert M.is_dispersive
    assert all_close(M(WL), xp.asarray([[1, 0], [0, n(WL)]]))

    # A mixed static + dispersive composition stays dispersive.
    M_sys = make_abcd_free_space(0.1, xp=xp) @ M
    assert M_sys.is_dispersive

    expected = make_abcd_free_space(0.1, xp=xp)(WL) @ M(WL)
    assert all_close(M_sys(WL), expected)

    # A constant callable is still dispersive.
    M = make_abcd_refractive_interface(lambda wl: 1.5, 1.0, 2.0, xp=xp)
    assert isinstance(M, RayTransferMatrix)
    assert M.is_dispersive

    # Dispersive derived scalars evaluate at the requested wavelength.
    M = make_abcd_refractive_interface(lambda wl: (wl + 2.25) ** 0.5, 1.0, 2.0, xp=xp)
    assert float(M.angular_magnification(WL)) == pytest.approx((WL + 2.25) ** 0.5)

def test_gaussian_beam_forward():
    w0 = 1e-3
    z = 0.5
    wavelength = 500e-9
    beam = GaussianBeam(z + 1j * math.pi * w0**2 / wavelength, wavelength)

    # Free-space propagation by a distance d forwards z by d and leaves zR unchanged.
    d = 0.3
    zR = beam.zR
    new_beam = make_abcd_free_space(d, xp=xp).forward(beam)
    assert new_beam is not beam
    assert float(new_beam.z) == pytest.approx(z + d)
    assert float(new_beam.zR) == pytest.approx(zR)
    assert float(beam.z) == pytest.approx(z)

    # A thin lens transforms q as q' = q f / (f - q).
    beam = GaussianBeam(z + 1j * math.pi * w0**2 / wavelength, wavelength)
    f = 2.0
    q = z + 1j * beam.zR
    new_beam = make_abcd_thin_lens(f, xp=xp).forward(beam)
    assert complex(new_beam.q) == pytest.approx((q * f) / (f - q))

    # Composing matrices is equivalent to forwarding through each consecutively.
    beam1 = make_abcd_free_space(d, xp=xp).forward(make_abcd_thin_lens(f, xp=xp).forward(beam))
    beam2 = (make_abcd_free_space(d, xp=xp) @ make_abcd_thin_lens(f, xp=xp)).forward(beam)
    assert complex(beam1.q) == pytest.approx(complex(beam2.q))

def test_gaussian_beam_backward():
    w0 = 1e-3
    z = 0.5
    wavelength = 500e-9
    beam = GaussianBeam(z + 1j * math.pi * w0**2 / wavelength, wavelength)

    # Backward through a free space of distance d moves z by -d.
    d = 0.3
    new_beam = make_abcd_free_space(d, xp=xp).backward(beam)
    assert float(new_beam.z) == pytest.approx(z - d)

    # Forward then backward returns (numerically) to the original beam.
    M = make_abcd_refractive_interface(1.0, 1.5, 2.0, xp=xp)
    back = M.backward(M.forward(beam))
    assert complex(back.q) == pytest.approx(complex(beam.q))

def test_gaussian_beam_forward_uses_beam_wavelength():
    beam = GaussianBeam(0.5 + 1j * math.pi * (1e-3)**2 / 500e-9, 500e-9)

    # Forwarding through 'distance' B = wavelength * 1e6 equals 0.5 at 500 nm.
    M = RayTransferMatrix(lambda wl: xp.asarray([[1.0, wl * 1e6], [0.0, 1.0]]))

    new_beam = M.forward(beam)
    assert float(new_beam.z) == pytest.approx(1.0)

def test_gaussian_beam_propagate():
    beam = GaussianBeam(0.5 + 1j * math.pi * (1e-3)**2 / 500e-9, 500e-9)

    # A raw 2x2 matrix propagates directly.
    new_beam = beam.propagate(xp.asarray([[1.0, 0.3], [0.0, 1.0]]))
    assert new_beam is not beam
    assert float(new_beam.z) == pytest.approx(0.8)

    # A callable is evaluated at the wavelength of the beam.
    new_beam = beam.propagate(lambda wl: xp.asarray([[1.0, wl * 1e6], [0.0, 1.0]]))
    assert float(new_beam.z) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        beam.propagate(xp.eye(3))
