from hcipy import *
import numpy as np
import os
import warnings
import itertools
import pytest
import functools

def check_against_reference(field_generator, diameter, baseline_name):
	fname = os.path.join(os.path.dirname(__file__), 'baseline_for_apertures/' + baseline_name + '.fits.gz')

	grid = make_uniform_grid(256, [diameter, diameter])
	field = evaluate_supersampled(field_generator, grid, 8)

	if os.path.isfile(fname):
		reference = read_fits(fname).ravel()

		assert np.allclose(field, reference)
	else:
		warnings.warn('Baseline aperture %s not available. Writing reference now...' % fname)

		if not os.path.exists(os.path.dirname(fname)):
			os.makedirs(os.path.dirname(fname))
		write_fits(field, fname)

def check_segmentation(aperture_function, segments=None):
	grid = make_uniform_grid(256, [1, 1])

	if segments is None:
		aperture, segments = aperture_function(normalized=True, return_segments=True)
	else:
		aperture = aperture_function

	aperture = evaluate_supersampled(aperture, grid, 2)
	segments = evaluate_supersampled(segments, grid, 2)

	aperture_from_segments = segments.linear_combination(np.ones(len(segments)))

	assert np.allclose(aperture, aperture_from_segments)

def check_aperture_against_reference(aperture_function, basename, diameter, options, segmented=False):
	keys = sorted(options.keys())
	vals = [value for key, value in sorted(options.items())]

	for option in itertools.product(*vals):
		fname = basename + '/pupil' + ''.join([o[1] for o in option])

		kwargs = dict(zip(keys, [o[0] for o in option]))
		aperture = aperture_function(**kwargs)

		check_against_reference(aperture, 1 if kwargs.get('normalized', False) else diameter, fname)

		if segmented:
			check_segmentation(*aperture_function(return_segments=True, **kwargs))

def test_regular_polygon_aperture():
	options = {
		'num_sides': [(6, '_hex'), (5, '_pent')],
		'angle': [(0, ''), (15, '_rotated')]
	}

	check_aperture_against_reference(functools.partial(regular_polygon_aperture, circum_diameter=1), 'polygon', 1, options)

def test_elliptical_aperture():
	options = {
		'diameters': [([0.5, 0.5], '_round'), ([1, 0.5], '_elongated')]
	}

	check_aperture_against_reference(elliptical_aperture, 'ellipse', 1, options)

def test_obstructed_circular_aperture():
	options = {
		'central_obscuration_ratio': [(0.1, '_small_obscuration'), (0.3, '_large_obscuration')],
		'num_spiders': [(3, '_3spiders'), (5, '_5spiders')],
		'spider_width': [(0.01, '_thinspiders'), (0.03, '_thickspiders')]
	}

	check_aperture_against_reference(functools.partial(make_obstructed_circular_aperture, pupil_diameter=1), 'obstructed_circular', 1, options)

def test_hexagonal_segmented_aperture():
	options = {
		'num_rings': [(3, '_3rings'), (5, '_5rings')],
		'segment_flat_to_flat': [(0.04, '_smallsegment'), (0.07, '_largesegment')],
		'gap_size': [(0.01, '_smallgap'), (0.02, '_largegap')],
		'starting_ring': [(0, '_withcenter'),  (2, '_withoutcenter')]
	}

	check_aperture_against_reference(make_hexagonal_segmented_aperture, 'hexagonal_segmented', 1, options, segmented=True)

def test_vlt_aperture():
	options = {
		'telescope': [('ut1', '_ut123'), ('ut2', '_ut123'), ('ut3', '_ut123'), ('antu', '_ut123'), ('kueyen', '_ut123'), ('melipal', '_ut123')],
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')]
	}

	check_aperture_against_reference(make_vlt_aperture, 'vlt', 8.1196, options)

	options = {
		'telescope': [('ut4', '_ut4'), ('yepun', '_ut4')],
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')],
		'with_M3_cover': [(False, ''), (True, '_with_M3_cover')]
	}

	check_aperture_against_reference(make_vlt_aperture, 'vlt', 8.1196, options)

	for telescope in ['ut1', 'ut2', 'ut3']:
		with pytest.warns(UserWarning, match='Using the M3 cover on a telescope other than UT4 is not realistic.'):
			aper = make_vlt_aperture(telescope=telescope, with_M3_cover=True)

	with pytest.raises(ValueError):
		aper = make_vlt_aperture(telescope='nonexistent_vlt_telescope')

def test_magellan_aperture():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')]
	}

	check_aperture_against_reference(make_magellan_aperture, 'magellan', 6.5, options)

def test_luvoir_a_aperture():
	check_against_reference(make_luvoir_a_aperture(), 15.0, 'luvoir_a/pupil')

	check_segmentation(make_luvoir_a_aperture)

@pytest.mark.slow
def test_luvoir_a_aperture_all():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')],
		'with_segment_gaps': [(True, ''), (False, '_without_segment_gaps')]
	}

	check_aperture_against_reference(make_luvoir_a_aperture, 'luvoir_a', 15, options)

def test_luvoir_a_lyot_stop():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')]
	}

	check_aperture_against_reference(make_luvoir_a_lyot_stop, 'luvoir_a_lyot', 15, options)

def test_luvoir_b_aperture():
	check_against_reference(make_luvoir_b_aperture(), 8.0, 'luvoir_b/pupil')
	check_segmentation(make_luvoir_b_aperture)

	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_segment_gaps': [(True, ''), (False, '_without_segment_gaps')]
	}
	check_aperture_against_reference(make_luvoir_b_aperture, 'luvoir_b', 8, options)

def test_hicat_aperture():
	check_against_reference(make_hicat_aperture(), 0.019725, 'hicat_pupil/pupil')

	check_segmentation(make_hicat_aperture)

@pytest.mark.slow
def test_hicat_aperture_all():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')],
		'with_segment_gaps': [(True, ''), (False, '_without_segment_gaps')]
	}

	check_aperture_against_reference(make_hicat_aperture, 'hicat_pupil', 0.019725, options)

def test_hicat_lyot_stop():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')]
	}

	check_aperture_against_reference(make_hicat_lyot_stop, 'hicat_lyot', 19.9e-3, options)

def test_elt_aperture():
	options = {
		'normalized': [(False, ''), (True, '_normalized')],
		'with_spiders': [(True, ''), (False, '_without_spiders')]
	}

	check_aperture_against_reference(make_elt_aperture, 'elt', 39.14634, options)
	
	check_segmentation(make_elt_aperture)