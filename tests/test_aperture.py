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
		warnings.warn('Baseline aperture not available. Writing reference now...')

		if not os.path.exists(os.path.dirname(fname)):
			os.makedirs(os.path.dirname(fname))
		write_fits(field, fname)

def check_segmentation(aperture_function):
	grid = make_uniform_grid(256, [1, 1])
	aperture, segments = aperture_function(normalized=True, return_segments=True)

	aperture = evaluate_supersampled(aperture, grid, 2)
	segments = evaluate_supersampled(segments, grid, 2)

	aperture_from_segments = segments.linear_combination(np.ones(len(segments)))
	
	assert np.allclose(aperture, aperture_from_segments)

def check_aperture_against_reference(aperture_function, basename, diameter, options):
	for option in itertools.product(*options.values()):
		fname = basename + '/pupil' + ''.join([o[1] for o in option])

		kwargs = dict(zip(options.keys(), [o[0] for o in option]))
		aperture = aperture_function(**kwargs)

		check_against_reference(aperture, 1 if kwargs.get('normalized', False) else diameter, fname)

def test_regular_polygon_aperture():
	options = {
		'num_sides': [(6, '_hex'), (5, '_pent')],
		'angle': [(0, ''), (15, '_rotated')]
	}

	check_aperture_against_reference(functools.partial(regular_polygon_aperture, circum_diameter=1), 'polygon', 1, options)

def test_obstructed_circular_aperture():
	options = {
		'central_obscuration_ratio': [(0.1, '_small_obscuration'), (0.3, '_large_obscuration')],
		'num_spiders': [(3, '_3spiders'), (5, '_5spiders')],
		'spider_width': [(0.01, '_thinspiders'), (0.03, '_thickspiders')]
	}

	check_aperture_against_reference(functools.partial(make_obstructed_circular_aperture, pupil_diameter=1), 'obstructed_circular', 1, options)

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
