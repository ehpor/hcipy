import numpy as np
from ..field import CartesianGrid, UnstructuredCoords, make_hexagonal_grid, Field
from .generic import *

def make_vlt_aperture():
	pass

def make_subaru_aperture():
	pass

def make_lbt_aperture():
	pass

def make_magellan_aperture(normalized=False, with_spiders=True):
	'''Make the Magellan aperture.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 6.5 meters.
	with_spiders: boolean
		If this is False, the spiders will be left out.

	Returns
	-------
	Field generator
		The Magellan aperture.
	'''
	pupil_diameter = 6.5 #m
	spider_width1 = 0.75 * 0.0254 #m
	spider_width2 = 1.5 * 0.0254 #m
	central_obscuration_ratio = 0.29
	spider_offset = [0,0.34] #m

	if normalized:
		spider_width1 /= pupil_diameter
		spider_width2 /= pupil_diameter
		spider_offset = [x / pupil_diameter for x in spider_offset]
		pupil_diameter = 1.0

	spider_offset = np.array(spider_offset)

	mirror_edge1 = (pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
	mirror_edge2 = (-pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
	mirror_edge3 = (pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))
	mirror_edge4 = (-pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))

	obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)

	if not with_spiders:
		return obstructed_aperture

	spider1 = make_spider(spider_offset, mirror_edge1, spider_width1)
	spider2 = make_spider(spider_offset, mirror_edge2, spider_width1)
	spider3 = make_spider(-spider_offset, mirror_edge3, spider_width2)
	spider4 = make_spider(-spider_offset, mirror_edge4, spider_width2)

	def func(grid):
		return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
	return func

def make_keck_aperture():
	pass

def make_luvoir_a_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, gap_padding=1, segment_transmissions=1, return_header=False, return_segments=False):
	'''
	This aperture changes frequently. This one is based on LUVOIR Apertures dimensions 
	from Matt Bolcar, LUVOIR lead engineer (as of 10 April 2019)
	Spiders and segment gaps can be included or excluded, and the transmission for each 
	of the segments can also be changed. Segements can be returned as well.

	Parameters
	----------
	normalized : boolean
		If this is True, the pupil diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	with_segment_gaps : boolean
		Include the gaps between individual segments in the aperture.
	gap_padding : scalar
		Arbitrary padding of gap size to represent gaps on smaller arrays - this effectively 
		makes the gaps larger and the segments smaller to preserve the same segment pitch.
	segment_transmissions : scalar or array_like
		The transmission for each of the segments. If this is a scalar, this transmission 
		will be used for all segments.
	return_header : boolean
		If this is True, a header will be returned giving all important values for the 
		created aperture for reference.
	return_segments : boolean
		If this is True, the segments will also be returned as a ModeBasis.
	
	Returns
	-------
	aperture : Field generator
		The LUVOIR A aperture.
	aperture_header : dict
		A dictionary containing all quantities used when making this aperture. Only returned if
		`return_header` is True.
	segments : list of Field generators
		The segments. Only returned when `return_segments` is True.
	'''
	pupil_diameter = 15.0 #m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
	pupil_inscribed = 13.5 #m actual inscribed diameter
	actual_segment_flat_diameter = 1.2225 #m actual segment flat-to-flat diameter
	actual_segment_gap = 0.006 #m actual gap size between segments
	spider_width = 0.150 #m actual strut size
	lower_spider_angle = 12.7 #deg angle at which lower spiders are offset from vertical
	spid_start = 0.30657 #m spider starting point distance from center of aperture

	segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
	if not with_segment_gaps:
		segment_gap = 0
	
	segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
	segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter
	
	num_rings = 6 #number of full rings of hexagons around central segment

	lower_spider_angle = 12.7 #deg spiders are upside-down 'Y' shaped; degree the lower two spiders are offset from vertical by this amount

	if not with_segment_gaps:
		segment_gap = 0

	aperture_header = {'TELESCOP':'LUVOIR A','D_CIRC': pupil_diameter, 'D_INSC':pupil_inscribed,\
						'SEG_F2F_D':actual_segment_flat_diameter,'SEG_GAP':actual_segment_gap, \
						'STRUT_W':spider_width,'STRUT_AN':lower_spider_angle,'NORM':normalized, \
						'SEG_TRAN':segment_transmissions,'GAP_PAD':gap_padding, 'STRUT_ST':spid_start, \
						'PROV':'MBolcar ppt 20180815'}
	
	if normalized:
		segment_circum_diameter /= pupil_diameter
		actual_segment_flat_diameter /= pupil_diameter
		actual_segment_gap /= pupil_diameter
		spider_width /= pupil_diameter
		spid_start /= pupil_diameter
		pupil_diameter = 1.0

	segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
	segment_positions = segment_positions.subset(circular_aperture(pupil_diameter * 0.98)) #corner clipping
	segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

	segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)
	
	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 90, spider_width)
		spider2 = make_spider_infinite([spid_start, 0], 270 - lower_spider_angle, spider_width)
		spider3 = make_spider_infinite([-spid_start, 0], 270 + lower_spider_angle, spider_width)

	segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=return_segments)

	if return_segments:
		segmented_aperture, segments = segmented_aperture

	def func(grid):
		res = segmented_aperture(grid)

		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid)

		return Field(res, grid)
	
	if with_spiders and return_segments:
		# Use function to return the lambda, to avoid incorrect binding of variables
		def segment_with_spider(segment):
			return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid)
		
		segments = [segment_with_spider(s) for s in segments]
	
	if return_header:
		if return_segments:
			return func, aperture_header, segments
		else:
			return func, aperture_header
	elif return_segments:
		return func, segments
	else:
		return func

def make_luvoir_a_lyot_stop(normalized=False, with_spiders=False, spider_oversize=1, inner_diameter_fraction=0.2, outer_diameter_fraction=0.9, return_header=False):
	'''Make a LUVOIR-A Lyot stop for the APLC coronagraph.

	Parameters
	----------
	normalized : boolean
		If this is True, the pupil diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	inner_diameter_fraction : scalar
		The fractional size of the circular central obstruction as fraction of the pupil diameter.
	outer_diameter_fraction : scalar
		The fractional size of the circular outer edge as fraction of the pupil diameter.
	spider_oversize : scalar
		The factor by which to oversize the spiders compared to the LUVOIR-A aperture spiders.
	return_header : boolean
		If this is True, a header will be returned giving all important values for the 
		created aperture for reference.

	Returns
	-------
	lyot_stop : Field generator
		A field generator for the Lyot stop.
	header : dict
		A dictionary containing all important values for the created aperture. Only returned
		if `return_header` is True.
	'''
	pupil_diameter = 15.0 #m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
	spider_width = 0.150 #m actual strut size
	lower_spider_angle = 12.7 #deg angle at which lower spiders are offset from vertical
	spid_start = 0.30657 #m spider starting point offset from center of aperture

	outer_D = pupil_diameter * outer_diameter_fraction
	inner_D = pupil_diameter * inner_diameter_fraction
	pad_spid_width = spider_width * spider_oversize

	lyot_reference_diameter = pupil_diameter

	ls_header = {'TELESCOP':'LUVOIR A', 'D_CIRC': pupil_diameter, 'D_INSC': 13.5,
				 'LS_ID': inner_diameter_fraction, 'LS_OD': outer_diameter_fraction, 'LS_REF_D': lyot_reference_diameter, 'NORM': normalized, 'STRUT_ST': spid_start}

	if with_spiders:
		ls_header['STRUT_W']  = spider_width
		ls_header['STRUT_AN'] = lower_spider_angle
		ls_header['STRUT_P']  = spider_oversize

	if normalized:
		outer_D /= pupil_diameter
		inner_D /= pupil_diameter
		pad_spid_width /= pupil_diameter
		spid_start /= pupil_diameter

	outer_diameter = circular_aperture(outer_D)
	central_obscuration = circular_aperture(inner_D)

	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 90, pad_spid_width)
		spider2 = make_spider_infinite([spid_start,0], 270 - lower_spider_angle, pad_spid_width)
		spider3 = make_spider_infinite([-spid_start,0], 270 + lower_spider_angle, pad_spid_width)

	def aper(grid):
		result = outer_diameter(grid) - central_obscuration(grid)

		if with_spiders:
			result *= spider1(grid) * spider2(grid) * spider3(grid)
		
		return result

	if return_header:
		return aper, ls_header
	
	return aper

def make_hicat_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, return_header=False, return_segments=False):
	'''Make the HiCAT P3 apodizer mask

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	with_segment_gaps : boolean
		Include the gaps between individual segments in the aperture.
	return_header : boolean
		If this is True, a header will be returned giving all important values for the 
		created aperture for reference.
	return_segments : boolean
		If this is True, the segments will also be returned as a list of Field generators.
	
	Returns
	-------
	aperture : Field generator
		The HiCAT aperture.
	header : dict
		A dictionary containing all important values for the created aperture. Only returned
		if `return_header` is True.
	segments : list of Field generators
		The segments. Only returned when `return_segments` is True.
	'''
	gamma_21 = 0.423
	gamma_31 = 1.008
	
	#P2 - Iris AO
	p2_irisao_segment_size = 1.4e-3 # m (note: point to point)
	p2_irisao_segment_side_length = p2_irisao_segment_size / 2
	p2_irisao_segment_gap_size = 12e-6 # m
	
	p2_irisao_distance_between_segments = p2_irisao_segment_side_length * np.sqrt(3)
	p2_irisao_segment_circumdiameter = (2 * p2_irisao_segment_side_length) - (2/np.sqrt(3)) * p2_irisao_segment_gap_size

	#P1 - Pupil Mask
	# Central segment
	p1_pupil_mask_central_segment_size = 3.600e-3 # m
	
	#P3 - Apodizer
	# Contour
	p3_apodizer_size = 19.725e-3 # m
	p2_apodizer_size = p3_apodizer_size * gamma_21 / gamma_31
	
	# Gap
	p3_apodizer_mask_gap_size = 0.090e-3 # m
	p3_irisao_segment_gap_size = p2_irisao_segment_gap_size * gamma_31 / gamma_21
	apodizer_mask_gap_oversize_factor_wrt_irisao = p3_apodizer_mask_gap_size / p3_irisao_segment_gap_size
	
	# Central segment
	p3_apodizer_mask_central_segment_size = 3.950e-3 # m
	p3_pupil_mask_central_segment_size = p1_pupil_mask_central_segment_size * gamma_31
	apodizer_mask_central_segment_oversize_factor_wrt_pupil_mask = p3_apodizer_mask_central_segment_size / p3_pupil_mask_central_segment_size
	p3_irisao_segment_size = p2_irisao_segment_size * gamma_31 / gamma_21

	# Spiders
	p3_apodizer_mask_spiders_thickness = 0.350e-3 # m
	
	header = {'TELESCOP':'HiCAT', 'P3_APOD': p3_apodizer_size, 'P3_CENT_SEG': p3_apodizer_mask_central_segment_size,
				'P3_GAP': p3_apodizer_mask_gap_size, 'P3_GAP_OVER': apodizer_mask_gap_oversize_factor_wrt_irisao,
				'P3_STRUT': p3_apodizer_mask_spiders_thickness, 'PROV': 'HiCAT spreadsheet'}
	
	p3_irisao_segment_circumdiameter = p2_irisao_segment_circumdiameter * gamma_31 / gamma_21
	p3_irisao_distance_between_segments = p2_irisao_distance_between_segments * gamma_31 / gamma_21
	p3_apodizer_segment_circumdiameter = p3_irisao_segment_circumdiameter + (-p3_apodizer_mask_gap_size + p3_irisao_segment_gap_size) * (2/np.sqrt(3))

	if normalized:
		p3_apodizer_segment_circumdiameter /= p3_apodizer_size
		p3_irisao_distance_between_segments /= p3_apodizer_size
		p3_apodizer_mask_central_segment_size /= p3_apodizer_size
		p3_apodizer_mask_spiders_thickness /= p3_apodizer_size
		p3_apodizer_size = 1

	segment = hexagonal_aperture(p3_apodizer_segment_circumdiameter, np.pi / 2)
	segment_positions = make_hexagonal_grid(p3_irisao_distance_between_segments, 3, False)
	segmentation = make_segmented_aperture(segment, segment_positions, return_segments=return_segments)

	if return_segments:
		segmentation, segments = segmentation
	
	segment = hexagonal_aperture(p3_apodizer_size / 7 / np.sqrt(3) * 2, np.pi / 2)
	distance_between_segments = p3_apodizer_size / 7
	segment_positions = make_hexagonal_grid(distance_between_segments, 3)
	contour = make_segmented_aperture(segment, segment_positions)
	
	central_segment = hexagonal_aperture(p3_apodizer_mask_central_segment_size, np.pi / 2)

	if return_segments:
		# Use function to return the lambda, to avoid incorrect binding of variables
		def segment_obstructed(segment):
			return lambda grid: segment(grid) * (contour(grid) - central_segment(grid))
		
		segments = [segment_obstructed(s) for s in segments]
	
	if with_spiders:
		spider1 = make_spider_infinite([0,0], 60, p3_apodizer_mask_spiders_thickness)
		spider2 = make_spider_infinite([0,0], 120, p3_apodizer_mask_spiders_thickness)
		spider3 = make_spider_infinite([0,0], -60, p3_apodizer_mask_spiders_thickness)
		spider4 = make_spider_infinite([0,0], -120, p3_apodizer_mask_spiders_thickness)
	
	if with_spiders and return_segments:
		# Use function to return the lambda, to avoid incorrect binding of variables
		def segment_with_spider(segment):
			return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
		
		segments = [segment_with_spider(s) for s in segments]
	
	def func(grid):
		res = contour(grid) - central_segment(grid)
		
		if with_segment_gaps:
			res *= segmentation(grid)
		
		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
		
		return Field(res, grid)
	
	if return_header:
		if return_segments:
			return func, header, segments
		else:
			return func, header
	elif return_segments:
		return func, segments
	else:
		return func

def make_hicat_lyot_stop(normalized=False, with_spiders=True, inner_diameter_fraction=0.2, outer_diameter_fraction=0.9, return_header=False):
	'''Make the HiCAT Lyot stop.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	inner_diameter_fraction : scalar
		The fractional size of the circular central obstruction as fraction of the pupil diameter.
	outer_diameter_fraction : scalar
		The fractional size of the circular outer edge as fraction of the pupil diameter.
	return_header : boolean
		If this is True, a header will be returned giving all important values for the 
		created aperture for reference.

	Returns
	-------
	lyot_stop : Field generator
		A field generator for the Lyot stop.
	header : dict
		A dictionary containing all important values for the created aperture. Only returned
		if `return_header` is True.
	'''
	gamma_21 = 0.423
	gamma_31 = 1.008
	gamma_51 = 0.979
	
	p2_irisao_segment_size = 1.4e-3 # m (note: point to point)
	p2_irisao_segment_side_length = p2_irisao_segment_size / 2
	p2_irisao_inscribed_circle_size = 10 * p2_irisao_segment_side_length
	p2_irisao_flat_to_flat_size = 14 * np.sqrt(3) / 2 * p2_irisao_segment_side_length
	p2_irisao_circumscribed_circle_size = np.sqrt(p2_irisao_flat_to_flat_size**2 + p2_irisao_segment_side_length**2)
	
	p3_apodizer_mask_central_segment_size = 3.950e-3 # m
	p3_apodizer_size = 19.725e-3 # m
	p5_apodizer_size = p3_apodizer_size * gamma_51 / gamma_31
	
	p5_lyot_stop_size = outer_diameter_fraction * p5_apodizer_size # m
	p5_irisao_inscribed_circle_size = p2_irisao_inscribed_circle_size * gamma_51 / gamma_21
	lyot_stop_mask_undersize_contour_wrt_inscribed_circle = p5_lyot_stop_size / p5_irisao_inscribed_circle_size

	p5_irisao_flat_to_flat_size = p2_irisao_flat_to_flat_size * gamma_51 / gamma_21
	p5_irisao_circumscribed_circle_size = p2_irisao_circumscribed_circle_size * gamma_51 / gamma_21

	# Central segment
	p5_lyot_stop_mask_central_segment_size = inner_diameter_fraction * p5_apodizer_size # m
	p5_apodizer_mask_central_segment_size = p3_apodizer_mask_central_segment_size * gamma_51 / gamma_31

	p5_irisao_segment_size = p2_irisao_segment_size * gamma_51 / gamma_21
	lyot_stop_mask_central_segment_oversize_factor_wrt_apodizer_mask = p5_lyot_stop_mask_central_segment_size / p5_apodizer_mask_central_segment_size
	
	# Spiders
	p5_lyot_stop_mask_spiders_thickness = 0.700e-3 # m
	lyot_stop_mask_spiders_thickness_ratio = p5_lyot_stop_mask_spiders_thickness / p5_irisao_circumscribed_circle_size

	if normalized:
		p5_lyot_stop_size /= p5_apodizer_size
		p5_lyot_stop_mask_central_segment_size /= p5_apodizer_size
		p5_lyot_stop_mask_spiders_thickness /= p5_apodizer_size
	
	central_obscuration = circular_aperture(p5_lyot_stop_mask_central_segment_size)
	outer_diameter = circular_aperture(p5_lyot_stop_size)
	
	header = {'TELESCOP':'HiCAT', 'P3_APOD': p3_apodizer_size, 'P3_CENT_SEG': p3_apodizer_mask_central_segment_size,
				'LS_CENT': p5_lyot_stop_mask_central_segment_size, 'LS_SIZE': p5_lyot_stop_size, 
				'P5_STRUT': p5_lyot_stop_mask_spiders_thickness}
	
	if with_spiders:
		spider1 = make_spider_infinite([0,0], 60, p5_lyot_stop_mask_spiders_thickness)
		spider2 = make_spider_infinite([0,0], 120, p5_lyot_stop_mask_spiders_thickness)
		spider3 = make_spider_infinite([0,0], -60, p5_lyot_stop_mask_spiders_thickness)
		spider4 = make_spider_infinite([0,0], -120, p5_lyot_stop_mask_spiders_thickness)

	def func(grid):
		res = outer_diameter(grid) - central_obscuration(grid)

		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)

		return Field(res, grid)
	
	if return_header:
		return func, header
	else:
		return func

def make_elt_aperture():
	pass
