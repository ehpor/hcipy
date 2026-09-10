import warnings

import numpy as np
import pytest

from hcipy.materials import (
    MaterialCatalog,
    AmbiguousMaterialError,
    make_sellmeier_index,
    make_sellmeier2_index,
    make_polynomial_index,
    make_rii_index,
    make_cauchy_index,
    make_gases_index,
    make_herzberger_index,
    make_retro_index,
    make_exotic_index
)

@pytest.fixture(scope='session')
def catalog():
    return MaterialCatalog()

def test_sellmeier():
    f = make_sellmeier_index([0, 0.6961663, 0.0684043, 0.4079426, 0.1162414, 0.8974794, 9.896161])
    assert f(0.5876e-6) == pytest.approx(1.45847, rel=1e-5)

def test_sellmeier2():
    f = make_sellmeier2_index([0, 1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653])
    assert f(0.5876e-6) == pytest.approx(1.51680, abs=1e-4)

def test_polynomial():
    f = make_polynomial_index([4, 0, 2])
    assert f(0.5e-6) == pytest.approx(2.0)

def test_rii():
    f = make_rii_index([4, 0, 2, 1, 2, 0, 2, 1, 2])
    assert f(0.5e-6) == pytest.approx(2.0)

def test_cauchy():
    f = make_cauchy_index([1.5, 0, -2])
    assert f(0.5e-6) == pytest.approx(1.5)

def test_gases():
    f = make_gases_index([0, 0, 1])
    assert f(0.5e-6) == pytest.approx(1.0)

def test_herzberger():
    f = make_herzberger_index([1.5, 0, 0, 0, 0])
    assert f(0.5e-6) == pytest.approx(1.5)

def test_retro():
    f = make_retro_index([0, 0, 1, 0])
    assert f(0.5e-6) == pytest.approx(1.0)

def test_exotic():
    f = make_exotic_index([4, 0, 1, 0, 1, 1])
    assert f(0.5e-6) == pytest.approx(2.0)

def test_material_formula(catalog):
    mat = catalog.get('N-BK7')

    assert mat.n(0.5876e-6) == pytest.approx(1.51680, abs=1e-4)
    assert mat.k(0.5876e-6) == pytest.approx(9.7525e-9)
    assert mat(0.5876e-6) == pytest.approx(1.51680 + 1j * 9.7525e-9, abs=1e-4)

    wl = np.array([0.4e-6, 0.6e-6])
    assert np.allclose(mat.n(wl), [mat.n(0.4e-6), mat.n(0.6e-6)])

def test_material_no_k(catalog):
    mat = catalog.get('Malitson', shelf='main', book='SiO2')

    assert mat.k(0.5e-6) == 0.0
    assert mat(0.5e-6) == pytest.approx(mat.n(0.5e-6))

def test_material_k_only(catalog):
    mat = catalog.get('Bosomworth-5K', shelf='main', book='BaF2')

    assert mat.k(3e-6) != 0.0
    with pytest.raises(ValueError):
        mat.n(5e-6)
    with pytest.raises(ValueError):
        mat(5e-6)

def test_material_tabulated(catalog):
    mat = catalog.get('Lane')

    assert mat.n(5e-6) == pytest.approx(1.3995)
    assert mat.k(3e-6) == pytest.approx(0.0369)

def test_material_properties(catalog):
    mat = catalog.get('N-BK7')

    assert mat.abbe() == pytest.approx(64.17, abs=0.1)
    assert mat.dispersion(0.4861327e-6, 0.6562725e-6) == pytest.approx(0.00805, abs=1e-3)
    assert mat.partial_dispersion(0.4861327e-6, 0.6562725e-6, 0.4861327e-6, 0.6562725e-6) == pytest.approx(1.0)

def test_page_info(catalog):
    info = catalog.get('N-BK7').page_info

    assert info['shelf'] == 'specs'
    assert info['book'] == 'SCHOTT-optical'
    assert info['page'] == 'N-BK7'
    assert info['wavelength_range'][0] == pytest.approx(0.3e-6)
    assert info['wavelength_range'][1] == pytest.approx(2.5e-6)

def test_get(catalog):
    mat = catalog.get('N-BK7')
    assert mat.page == 'N-BK7'

def test_get_ambiguous(catalog):
    with pytest.raises(AmbiguousMaterialError) as exc:
        catalog.get('Malitson')
    assert len(exc.value.candidates) > 1

    mat = catalog.get('Malitson', shelf='main', book='SiO2')
    assert mat.book == 'SiO2'
    assert mat.n(0.5876e-6) == pytest.approx(1.45846, rel=1e-5)

def test_get_narrowed_miss(catalog):
    with pytest.raises(KeyError):
        catalog.get('N-BK7', book='OHARA-optical')

def test_get_missing(catalog):
    with pytest.raises(KeyError):
        catalog.get('N-BK4')

def test_search(catalog):
    results = catalog.search('BK7')
    assert len(results) > 0
    assert all('BK7' in m.page or 'BK7' in m.book for m in results)

    assert catalog.search('nope') == []

def test_search_all(catalog):
    results = catalog.search()

    assert 0 < len(results) <= len(catalog)
    assert all(m.page for m in results)

def test_bad_coefficient_counts():
    with pytest.raises(ValueError):
        make_sellmeier_index([0, 0.6961663, 0.0684043, 0.4079426])

    with pytest.raises(ValueError):
        make_herzberger_index([1.5, 0, 0])

    with pytest.raises(ValueError):
        make_exotic_index([4, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        make_rii_index([4, 0, 2, 1, 2])

def test_wavelength_range_warning(catalog):
    mat = catalog.get('N-BK7')

    with pytest.warns(UserWarning, match='outside its valid wavelength range'):
        mat.n(3e-6)

    with pytest.warns(UserWarning, match='outside its valid wavelength range'):
        mat.k(3e-6)

    with pytest.warns(UserWarning, match='outside its valid wavelength range'):
        mat(3e-6)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mat.n(0.6e-6)
        mat.k(0.6e-6)
        mat(0.6e-6)
        mat.n(3e-6, allow_extrapolation=True)
        mat.k(3e-6, allow_extrapolation=True)
        mat(3e-6, allow_extrapolation=True)

def test_explicit_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        MaterialCatalog(tmp_path / 'nonexistent')
