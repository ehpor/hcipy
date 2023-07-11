from hcipy import *
import numpy as np
import os
import warnings
import pytest
import functools

def check_aperture(aperture_function, diameter, name, check_normalization=False, check_segmentation=False, **aperture_args):
    fname = os.path.join(os.path.dirname(__file__), 'baseline_for_apertures/' + name + '.fits.gz')

    grid = make_uniform_grid(256, [diameter, diameter])
    field = evaluate_supersampled(aperture_function(**aperture_args), grid, 8)

    if check_normalization:
        grid_norm = grid.scaled(1 / diameter)
        field_norm = evaluate_supersampled(aperture_function(normalized=True, **aperture_args), grid_norm, 8)

        assert np.allclose(field, field_norm)

    if check_segmentation:
        aperture, segments = aperture_function(**aperture_args, return_segments=True)

        aperture = evaluate_supersampled(aperture, grid, 1)
        segments = evaluate_supersampled(segments, grid, 1)

        aperture_from_segments = segments.linear_combination(np.ones(len(segments)))

        assert np.allclose(aperture, aperture_from_segments)

    if os.path.isfile(fname):
        reference = read_fits(fname).ravel()

        assert np.allclose(field, reference)
    else:
        warnings.warn('Baseline aperture %s not available. Writing reference now...' % fname)

        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        write_fits(field, fname)

def test_regular_polygon_aperture():
    pupil_diameter = 1

    for num_sides in [6, 5]:
        for angle in [0, 15]:
            name = 'polygon/pupil'
            name += '_rotated' if angle != 0 else ''
            name += '_hex' if num_sides == 6 else '_pent'

            check_aperture(
                make_regular_polygon_aperture, pupil_diameter, name,
                circum_diameter=pupil_diameter, num_sides=num_sides, angle=angle
            )

def test_circular_aperture():
    pupil_diameter = 1

    for diameter in [1, 0.5]:
        name = 'circular/pupil'
        name += '_small' if diameter < 0.7 else '_large'

        # Use a functools.partial() here to avoid the name collision of the diameter argument.
        check_aperture(functools.partial(make_circular_aperture, diameter=diameter), pupil_diameter, name)

def test_rectangular_aperture():
    pupil_diameter = 1

    for size in [[0.5, 0.5], [1, 0.5]]:
        name = 'rectangular/pupil'
        name += '_square' if size[0] == size[1] else '_elongated'

        check_aperture(make_rectangular_aperture, pupil_diameter, name, size=size)

def test_elliptical_aperture():
    pupil_diameter = 1

    for diameters in [[0.5, 0.5], [1, 0.5]]:
        name = 'ellipse/pupil'
        name += '_round' if diameters[0] == diameters[1] else '_elongated'

        check_aperture(make_elliptical_aperture, pupil_diameter, name, diameters=diameters)

def test_obstructed_circular_aperture():
    pupil_diameter = 1

    for central_obscuration_ratio in [0.1, 0.3]:
        for num_spiders in [3, 5]:
            for spider_width in [0.01, 0.03]:
                name = 'obstructed_circular/pupil'
                name += '_small_obscuration' if central_obscuration_ratio < 0.2 else '_large_obscuration'
                name += f'_{num_spiders}spiders'
                name += '_thinspiders' if spider_width < 0.02 else '_thickspiders'

                check_aperture(
                    make_obstructed_circular_aperture, pupil_diameter, name,
                    pupil_diameter=pupil_diameter, central_obscuration_ratio=central_obscuration_ratio, num_spiders=num_spiders, spider_width=spider_width
                )

@pytest.mark.parametrize('num_rings', [3, 5])
@pytest.mark.parametrize('segment_flat_to_flat', [0.04, 0.07])
@pytest.mark.parametrize('gap_size', [0.01, 0.02])
@pytest.mark.parametrize('starting_ring', [0, 2])
def test_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size, starting_ring):
    name = 'hexagonal_segmented/pupil'
    name += '_smallgap' if gap_size < 0.015 else '_largegap'
    name += f'_{num_rings}rings'
    name += '_smallsegment' if segment_flat_to_flat < 0.05 else '_largesegment'
    name += '_withcenter' if starting_ring == 0 else '_withoutcenter'

    check_aperture(
        make_hexagonal_segmented_aperture, 1, name,
        check_segmentation=True,
        num_rings=num_rings, segment_flat_to_flat=segment_flat_to_flat, gap_size=gap_size, starting_ring=starting_ring
    )

@pytest.mark.parametrize('telescope', ['ut1', 'ut2', 'ut3', 'antu', 'kueyen', 'melipal'])
@pytest.mark.parametrize('with_spiders', [True, False])
def test_vlt_ut_123_aperture(telescope, with_spiders):
    name = 'vlt/pupil_ut123'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_vlt_aperture, 8.0, name,
        check_normalization=True,
        check_segmentation=True,
        telescope=telescope, with_spiders=with_spiders
    )

@pytest.mark.parametrize('telescope', ['ut4', 'yepun'])
@pytest.mark.parametrize('with_spiders', [True, False])
@pytest.mark.parametrize('with_M3_cover', [True, False])
def test_vlt_ut_4_aperture(telescope, with_spiders, with_M3_cover):
    name = 'vlt/pupil_ut4'
    name += '_without_spiders' if not with_spiders else ''
    name += '_with_M3_cover' if with_M3_cover else ''

    check_aperture(
        make_vlt_aperture, 8.1196, name,
        check_normalization=True,
        check_segmentation=True,
        telescope=telescope, with_spiders=with_spiders, with_M3_cover=with_M3_cover
    )

def test_invalid_vlt_apertures():
    for telescope in ['ut1', 'ut2', 'ut3']:
        with pytest.warns(UserWarning, match='Using the M3 cover on a telescope other than UT4 is not realistic.'):
            make_vlt_aperture(telescope=telescope, with_M3_cover=True)

    with pytest.raises(ValueError):
        make_vlt_aperture(telescope='nonexistent_vlt_telescope')


@pytest.mark.parametrize('with_spiders', [True, False])
def test_vlti_aperture(with_spiders):
    name = 'vlti/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_vlti_aperture, 125.0, name,
        check_normalization=False, check_segmentation=True,
        with_spiders=with_spiders, zenith_angle=0.0, azimuth=0.0
    )

    name = 'vlti/pupil_non_zenith'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_vlti_aperture, 125.0, name,
        check_normalization=False, check_segmentation=True,
        with_spiders=with_spiders, zenith_angle=np.pi / 4, azimuth=np.pi / 4
    )

@pytest.mark.parametrize('with_spiders', [True, False])
def test_magellan_aperture(with_spiders):
    name = 'magellan/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_magellan_aperture, 6.5, name,
        check_normalization=True,
        with_spiders=with_spiders
    )

@pytest.mark.parametrize('with_spiders', [True, False])
def test_hale_aperture(with_spiders):
    name = 'hale/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_hale_aperture, 5.08, name,
        check_normalization=True,
        with_spiders=with_spiders
    )

@pytest.mark.parametrize('with_spiders', [True, False])
@pytest.mark.parametrize('with_segment_gaps', [True, False])
def test_luvoir_a_aperture(with_spiders, with_segment_gaps):
    name = 'luvoir_a/pupil'
    name += '_without_spiders' if not with_spiders else ''
    name += '_without_segment_gaps' if not with_segment_gaps else ''

    check_aperture(
        make_luvoir_a_aperture, 15, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders, with_segment_gaps=with_segment_gaps
    )

def test_luvoir_a_lyot_stop():
    for with_spiders in [True, False]:
        for spider_oversize in [1, 3]:
            for outer_diameter_fraction in [0.8, 0.9]:
                name = 'luvoir_a_lyot/pupil'
                name += '_withoutspiders' if not with_spiders else ''
                name += f'_{spider_oversize}spideroversize'
                name += f'_{int(outer_diameter_fraction * 100)}od'

                check_aperture(
                    make_luvoir_a_lyot_stop, 15, name,
                    check_normalization=True,
                    with_spiders=with_spiders, spider_oversize=spider_oversize, outer_diameter_fraction=outer_diameter_fraction
                )

@pytest.mark.parametrize('with_segment_gaps', [True, False])
def test_luvoir_b_aperture(with_segment_gaps):
    name = 'luvoir_b/pupil'
    name += '_without_segment_gaps' if not with_segment_gaps else ''

    check_aperture(
        make_luvoir_b_aperture, 8, name,
        check_normalization=True, check_segmentation=True,
        with_segment_gaps=with_segment_gaps
    )

@pytest.mark.parametrize('with_spiders', [True, False])
@pytest.mark.parametrize('with_segment_gaps', [True, False])
def test_hicat_aperture(with_spiders, with_segment_gaps):
    name = 'hicat_pupil/pupil'
    name += '_without_spiders' if not with_spiders else ''
    name += '_without_segment_gaps' if not with_segment_gaps else ''

    check_aperture(
        make_hicat_aperture, 0.019725, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders, with_segment_gaps=with_segment_gaps
    )

@pytest.mark.parametrize('with_spiders', [True, False])
def test_hicat_lyot(with_spiders):
    name = 'hicat_lyot/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_hicat_lyot_stop, 0.01915751488095238, name,
        check_normalization=True,
        with_spiders=with_spiders
    )

@pytest.mark.parametrize('with_spiders', [True, pytest.param(False, marks=pytest.mark.slow)])
def test_elt_aperture(with_spiders):
    name = 'elt/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_elt_aperture, 39.14634, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders
    )

@pytest.mark.parametrize('with_spiders', [True, pytest.param(False, marks=pytest.mark.slow)])
def test_tmt_aperture(with_spiders):
    name = 'tmt/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_tmt_aperture, 30.0, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders
    )

@pytest.mark.parametrize('with_spiders', [True, False])
def test_gmt_aperture(with_spiders):
    name = 'gmt/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_gmt_aperture, 25.448, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders
    )

def test_habex_aperture():
    name = 'habex/pupil'

    check_aperture(make_habex_aperture, 4.0, name)

@pytest.mark.parametrize('with_spiders', [True, False])
@pytest.mark.parametrize('with_pads', [True, False])
def test_hst_aperture(with_spiders, with_pads):
    name = 'hst/pupil'
    name += '_without_spiders' if not with_spiders else ''
    name += '_without_pads' if not with_pads else ''

    check_aperture(
        make_hst_aperture, 2.4, name,
        check_normalization=True,
        with_spiders=with_spiders, with_pads=with_pads
    )

@pytest.mark.parametrize('with_spiders', [True, False])
def test_jwst_aperture(with_spiders):
    name = 'jwst/pupil'
    name += '_without_spiders' if not with_spiders else ''

    check_aperture(
        make_jwst_aperture, 6.603464, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders
    )

def test_shifted_aperture():
    grid = make_pupil_grid(256, 2.0)
    aperture1 = make_circular_aperture(1.0, center=[0.25, 0.25])(grid)
    aperture2 = make_shifted_aperture(make_circular_aperture(1.0), np.array([0.25, 0.25]))(grid)

    assert np.allclose(aperture1, aperture2)

@pytest.mark.parametrize('with_spiders', [True, False])
@pytest.mark.parametrize('with_segment_gaps', [True, False])
def test_keck_aperture(with_spiders, with_segment_gaps):
    name = 'keck/pupil'
    name += '_without_spiders' if not with_spiders else ''
    name += '_without_segment_gaps' if not with_segment_gaps else ''

    check_aperture(
        make_keck_aperture, 10.95, name,
        check_normalization=True, check_segmentation=True,
        with_spiders=with_spiders, with_segment_gaps=with_segment_gaps
    )
