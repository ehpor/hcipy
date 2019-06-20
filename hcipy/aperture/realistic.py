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

def make_luvoir_a_aperture(gap_padding = 5, normalized=True, with_spiders=True, with_segment_gaps=True, segment_transmissions=1, header = True, return_segment_positions=False):
    
	'''Make the LUVOIR A aperture and Lyot Stop.

	This aperture changes frequently. This one is based on LUVOIR Apertures dimensions 
	from Matt Bolcar, LUVOIR lead engineer (as of 10 April 2019)
	Spiders and segment gaps can be included or excluded, and the transmission for each 
	of the segments can also be changed.

	Parameters
	----------
	gap_padding: float
		arbitratry padding of gap size to represent gaps on smaller arrays - effectively 
		makes the gaps larger and the segments smaller to preserve the same segment pitch 
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	with_segment_gaps : boolean
		Include the gaps between individual segments in the aperture.
	segment_transmissions : scalar or array_like
		The transmission for each of the segments. If this is a scalar, this transmission 
		will be used for all segments.
	return_segment_positions : boolean
		Will also return the segment positions as segment_position
	
	Returns
	-------
	Field generator
		The LUVOIR A aperture.
	
	aperture_header: dictionary
		dictionary of keywords to build aperture fits header
	'''
	
	pupil_diameter = 15.0 #m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
	pupil_inscribed = 13.5 #m actual inscribed diameter
	actual_segment_flat_diameter = 1.2225 #m actual segment flat-to-flat diameter
	actual_segment_gap=0.006 #m actual gap size between segments
	spider_width=0.150 #m actual strut size
	lower_spider_angle = 12.7 #deg angle at which lower spiders are offset from vertical
	spid_start = 0.30657 #m spider starting point distance from center of aperture

	segment_gap =  actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
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

	hexagon = hexagonal_aperture(segment_circum_diameter)
	def segment(grid):
		return hexagon(grid.rotated(np.pi/2))
	
	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 90, spider_width)
		spider2 = make_spider_infinite([spid_start, 0], 270 - lower_spider_angle, spider_width)
		spider3 = make_spider_infinite([-spid_start, 0], 270 + lower_spider_angle, spider_width)

	segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions)

	def func(grid):
		res = segmented_aperture(grid)

		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid)

		return Field(res, grid)
	
	blah = 1

	if return_segment_positions:
		return func, aperture_header, segment_positions
	return func, aperture_header
	

def make_luvoir_a_lyot_stop(ls_id, ls_od, lyot_ref_diameter, spid_oversize=1, normalized=True, spiders=False, header = False):

	pupil_diameter=15.0 #m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter

	spider_width=0.150 #m actual strut size
	lower_spider_angle = 12.7 #deg angle at which lower spiders are offset from vertical
	spid_start = 0.30657 #m spider starting point offset from center of aperture


	outer_D = lyot_ref_diameter*ls_od
	inner_D = lyot_ref_diameter*ls_id
	pad_spid_width = spider_width * spid_oversize

	ls_header = {'TELESCOP':'LUVOIR A','D_CIRC': pupil_diameter,'D_INSC':13.5,
				 'LS_ID':ls_id, 'LS_OD':ls_od,'LS_REF_D':lyot_ref_diameter, 'NORM':normalized, 'STRUT_ST':spid_start}

	if spiders:
		ls_header['STRUT_W']  = spider_width
		ls_header['STRUT_AN'] = lower_spider_angle
		ls_header['STRUT_P']  = spid_oversize

	if normalized:
		outer_D /= pupil_diameter
		inner_D /= pupil_diameter
		pad_spid_width /= pupil_diameter
		spid_start /= pupil_diameter

	outer_diameter = circular_aperture(outer_D)
	central_obscuration = circular_aperture(inner_D)

	if spiders:
		spider1 = make_spider_infinite([0, 0], 90, pad_spid_width)
		spider2 = make_spider_infinite([spid_start,0], 270 - lower_spider_angle, pad_spid_width)
		spider3 = make_spider_infinite([-spid_start,0], 270 + lower_spider_angle, pad_spid_width)

	def aper(grid):
		tmp_result = (outer_diameter(grid) - central_obscuration(grid)) 
		if spiders:
			tmp_result *= spider1(grid) * spider2(grid) * spider3(grid)
		return tmp_result    
	if header:
		return aper, ls_header
	
	return aper

def make_hicat_aperture(with_spiders=True, with_segment_gaps=True):
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

	Returns
	-------
	Field generator
		The HiCAT aperture.
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

	#Spiders
	p3_apodizer_mask_spiders_thickness = 0.350e-3 # m
	
	header = {'TELESCOP':'HiCAT','P3_APOD':p3_apodizer_size,'P3_CENT_SEG':p3_apodizer_mask_central_segment_size,
				'P3_GAP':p3_apodizer_mask_gap_size,'P3_GAP_OVER':apodizer_mask_gap_oversize_factor_wrt_irisao,
				'P3_STRUT':p3_apodizer_mask_spiders_thickness, 'PROV':'HiCAT spreadsheet'}
	
	p3_irisao_segment_circumdiameter = p2_irisao_segment_circumdiameter * gamma_31 / gamma_21
	p3_irisao_distance_between_segments = p2_irisao_distance_between_segments * gamma_31 / gamma_21
	p3_apodizer_segment_circumdiameter = p3_irisao_segment_circumdiameter + (-p3_apodizer_mask_gap_size + p3_irisao_segment_gap_size) * (2/np.sqrt(3))

	segment = hexagonal_aperture(p3_apodizer_segment_circumdiameter, np.pi/2)
	segment_positions = make_hexagonal_grid(p3_irisao_distance_between_segments, 3, False)
	segmentation = make_segmented_aperture(segment, segment_positions)
	
	segment = hexagonal_aperture(p3_apodizer_size / 7 / np.sqrt(3) * 2, np.pi / 2)
	distance_between_segments = p3_apodizer_size / 7
	segment_positions = make_hexagonal_grid(distance_between_segments, 3)
	contour = make_segmented_aperture(segment, segment_positions)
	
	central_segment = hexagonal_aperture(p3_apodizer_mask_central_segment_size, np.pi / 2)
	
	if with_spiders:
		spider1 = make_spider_infinite([0,0], 60, p3_apodizer_mask_spiders_thickness)
		spider2 = make_spider_infinite([0,0], 120, p3_apodizer_mask_spiders_thickness)
		spider3 = make_spider_infinite([0,0], -60, p3_apodizer_mask_spiders_thickness)
		spider4 = make_spider_infinite([0,0], -120, p3_apodizer_mask_spiders_thickness)
	
	def func(grid):
		res = (contour(grid) - central_segment(grid)) 
		
		if with_segment_gaps:
			res*= segmentation(grid)
		
		if with_spiders:
			res*= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
		
		return Field(res, grid)
		
	return func, header

def make_hicat_lyot_stop(ls_id, ls_od, with_spiders=False):
	'''Make the HiCAT Lyot stop.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.

	Returns
	-------
	Field generator
		The HiCAT Lyot stop.
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
	
	p5_lyot_stop_size = ls_od # m
	p5_irisao_inscribed_circle_size = p2_irisao_inscribed_circle_size * gamma_51 / gamma_21
	lyot_stop_mask_undersize_contour_wrt_inscribed_circle = p5_lyot_stop_size / p5_irisao_inscribed_circle_size

	p5_irisao_flat_to_flat_size = p2_irisao_flat_to_flat_size * gamma_51 / gamma_21
	p5_irisao_circumscribed_circle_size = p2_irisao_circumscribed_circle_size * gamma_51 / gamma_21

	# Central segment
	p5_lyot_stop_mask_central_segment_size = ls_id # m
	p5_apodizer_mask_central_segment_size = p3_apodizer_mask_central_segment_size * gamma_51 / gamma_31

	p5_irisao_segment_size = p2_irisao_segment_size * gamma_51 / gamma_21
	lyot_stop_mask_central_segment_oversize_factor_wrt_apodizer_mask = p5_lyot_stop_mask_central_segment_size / p5_apodizer_mask_central_segment_size
	
	# Spiders
	p5_lyot_stop_mask_spiders_thickness = 0.700e-3 # m
	lyot_stop_mask_spiders_thickness_ratio = p5_lyot_stop_mask_spiders_thickness / p5_irisao_circumscribed_circle_size
	
	central_obscuration = circular_aperture(p5_lyot_stop_mask_central_segment_size)
	outer_diameter = circular_aperture(p5_lyot_stop_size)
	
	header = {'TELESCOP':'HiCAT','P3_APOD':p3_apodizer_size,'P3_CENT_SEG':p3_apodizer_mask_central_segment_size,
				'LS_CENT':p5_lyot_stop_mask_central_segment_size, 'LS_SIZE':p5_lyot_stop_size, 
				'P5_STRUT':p5_lyot_stop_mask_spiders_thickness}
	
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
		
	return func,header

def make_elt_aperture():
	pass
