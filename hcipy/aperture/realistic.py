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

def make_luvoir_a_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, segment_transmissions=1, return_segment_positions=False):
	'''Make the LUVOIR A aperture.

	This aperture changes frequently. This one is based on [1]_. Spiders and segment gaps 
	can be included or excluded, and the transmission for each of the segments can also be changed.
	
	.. [1] Ruane et al. 2018 "Fast linearized coronagraph optimizer (FALCO) IV: coronagraph design 
		survey for obstructed and segmented apertures." Space Telescopes and Instrumentation 2018: 
		Optical, Infrared, and Millimeter Wave. Vol. 10698. International Society for Optics and Photonics, 2018.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	with_segment_gaps : boolean
		Include the gaps between individual segments in the aperture.
	segment_transmissions : scalar or array_like
		The transmission for each of the segments. If this is a scalar, this transmission will
		be used for all segments.
	return_segment_positions : boolean
		If this is True, the centers of each of the segments will get returned as well.
	
	Returns
	-------
	Field generator
		The LUVOIR A aperture.
	CartesianGrid
		The segment positions. Only returned when `return_segment_positions` is True.
	'''
	pupil_diameter = 15.0 # m
	segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 12
	num_rings = 6
	segment_gap = 0.001 * pupil_diameter
	spider_width = 0.005 * pupil_diameter
	spider_relative_offset = 0.9 # as estimated from the paper, not an actual value

	if not with_segment_gaps:
		segment_gap = 0

	if normalized:
		segment_circum_diameter /= pupil_diameter
		segment_gap /= pupil_diameter
		spider_width /= pupil_diameter
		pupil_diameter = 1.0

	segment_positions = make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
	segment_positions = segment_positions.subset(circular_aperture(pupil_diameter * 0.98))
	segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

	hexagon = hexagonal_aperture(segment_circum_diameter - segment_gap)
	def segment(grid):
		return hexagon(grid.rotated(np.pi/2))
	
	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 90, spider_width)

		p1 = np.array([-segment_circum_diameter * 0.5 * spider_relative_offset, 0])
		p2 = np.array([p1[0], -np.sqrt(3) * segment_circum_diameter + (segment_circum_diameter * 0.5 * (1 - spider_relative_offset)) * np.sqrt(3)])
		spider2 = make_spider(p1, p2, spider_width)

		p3 = p2 - np.array([pupil_diameter / 2, pupil_diameter * np.sqrt(3) / 2])
		spider3 = make_spider(p2, p3, spider_width)

		p4 = p1 * np.array([-1, 1])
		p5 = p2 * np.array([-1, 1])
		p6 = p3 * np.array([-1, 1])

		spider4 = make_spider(p4, p5, spider_width)
		spider5 = make_spider(p5, p6, spider_width)
	
	segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions)

	def func(grid):
		res = segmented_aperture(grid)
		
		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * spider5(grid)

		return Field(res, grid)

	if return_segment_positions:
			return func, segment_positions

	return func

def make_hicat_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, segment_transmissions=1, return_segment_positions=False):
	'''Make the HiCAT pupil mask.

	This function is a WIP. It should NOT be used for actual designs. Current pupil should be taken as 
	representative only.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
	with_spiders : boolean
		Include the secondary mirror support structure in the aperture.
	with_segment_gaps : boolean
		Include the gaps between individual segments in the aperture.
	segment_transmissions : scalar or array_like
		The transmission for each of the segments. If this is a scalar, this transmission will
		be used for all segments.
	return_segment_positions : boolean
		If this is True, the centers of each of the segments will get returned as well.
	
	Returns
	-------
	Field generator
		The HiCAT aperture.
	CartesianGrid
		The segment positions. Only returned when `return_segment_positions` is True.
	'''
	pupil_diameter = 0.019725 # m
	segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 7
	num_rings = 3
	segment_gap = 90e-6
	spider_width = 350e-6

	if not with_segment_gaps:
		segment_gap = 0

	if normalized:
		segment_circum_diameter /= pupil_diameter
		segment_gap /= pupil_diameter
		spider_width /= pupil_diameter
		pupil_diameter = 1.0

	segment_positions = make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
	segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

	hexagon = hexagonal_aperture(segment_circum_diameter - segment_gap)
	def segment(grid):
		return hexagon(grid.rotated(np.pi/2))

	segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions)

	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 60, spider_width)
		spider2 = make_spider_infinite([0, 0], 120, spider_width)
		spider3 = make_spider_infinite([0, 0], 240, spider_width)
		spider4 = make_spider_infinite([0, 0], 300, spider_width)

	def func(grid):
		res = segmented_aperture(grid)

		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
		
		return Field(res, grid)

	if return_segment_positions:
			return func, segment_positions

	return func

def make_hicat_lyot_stop(normalized=False, with_spiders=True):
	'''Make the HiCAT Lyot stop.

	This function is a WIP. It should NOT be used for actual designs. Current Lyot stop should be taken as 
	representative only.

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
	pupil_diameter = 19.9e-3
	lyot_outer = 15.9e-3
	lyot_inner = 6.8e-3
	spider_width = 700e-6

	if normalized:
		lyot_inner /= pupil_diameter
		lyot_outer /= pupil_diameter
		spider_width /= pupil_diameter

	aperture = circular_aperture(lyot_outer)
	obscuration = circular_aperture(lyot_inner)

	if with_spiders:
		spider1 = make_spider_infinite([0, 0], 60, spider_width)
		spider2 = make_spider_infinite([0, 0], 120, spider_width)
		spider3 = make_spider_infinite([0, 0], 240, spider_width)
		spider4 = make_spider_infinite([0, 0], 300, spider_width)
	
	def func(grid):
		res = aperture(grid) - obscuration(grid)

		if with_spiders:
			res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
		
		return Field(res, grid)
	return func

def make_elt_aperture():
	pass
