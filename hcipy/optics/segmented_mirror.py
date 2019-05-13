import os
import numpy as np
import matplotlib.pyplot as plt

from .optical_element import OpticalElement
from ..aperture import circular_aperture, hexagonal_aperture, make_segmented_aperture
from ..field import Field, make_hexagonal_grid, make_pupil_grid, evaluate_supersampled
from ..io import write_fits
from ..plotting import imshow_field

pupil_size = 1024
PUP_DIAMETER = 15.   # m

def get_atlast_aperture(normalized=False, with_segment_gaps=True, segment_transmissions=1, return_segment_positions=False):
	"""Make the ATLAST/HiCAT pupil mask.

	This function is a copy of make_hicat_aperture(), except that it also returns the segment positions.

	Parameters
	----------
	normalized : boolean
		If this is True, the outer diameter will be scaled to 1. Otherwise, the
		diameter of the pupil will be 15.0 meters.
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
		The ATLAST aperture.
	CartesianGrid
		The segment positions. Only returned when `return_segment_positions` is True.
	"""
	pupil_diameter = 15 # m
	segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 7
	num_rings = 3
	segment_gap = 0.01 # m

	if not with_segment_gaps:
		segment_gap = 0

	if normalized:
		segment_circum_diameter /= pupil_diameter
		segment_gap /= pupil_diameter
		pupil_diameter = 1.0

	segment_positions = make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
	segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

	hexagon = hexagonal_aperture(segment_circum_diameter - segment_gap, np.pi / 2)

	def segment(grid):
		return hexagon(grid)

	segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions)

	def func(grid):
		res = segmented_aperture(grid)

		return Field(res, grid)

	if return_segment_positions:
		return func, segment_positions
	
	return func

class SegmentedMirror(OpticalElement):
	"""A segmented mirror from a segmented aperture.

	Parameters:
	----------
	aperture : Field
		The segmented aperture of the mirror.
	seg_pos : CartesianGrid(UnstructuredCoords)
		Segment positions of the aperture.
	"""

	def __init__(self, aperture, seg_pos):
		self.aperture = aperture
		self.segnum = len(seg_pos.x)
		self.segmentlist = np.arange(1, self.segnum + 1)
		self._coef = np.zeros((self.segnum, 3))
		self.seg_pos = seg_pos
		self.input_grid = aperture.grid
		self._last_npix = np.nan  # see _setup_grids for this
		self._surface = None

	def forward(self, wavefront):
		"""Propagate a wavefront through the segmented mirror.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.

		Returns
		-------
		Wavefront
			The reflected wavefront.
		"""
		wf = wavefront.copy()
		wf.electric_field *= np.exp(2j * self.surface * wavefront.wavenumber)
		return wf

	def backward(self, wavefront):
		"""Propagate a wavefront backwards through the deformable mirror.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.

		Returns
		-------
		Wavefront
			The reflected wavefront.
		"""
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-2j * self.surface * wavefront.wavenumber)
		return wf

	@property
	def surface(self):
		""" The surface of the segmented mirror in meters, the full surface as a Field.
		"""
		if self._surface is None:
			self._surface = self.apply_coef()
		return self._surface

	@property
	def coef(self):
		""" The surface shape of the deformable mirror, in meters and radians; PTT segment coefficients.
		"""
		self._surface = None
		return self._coef

	def flatten(self):
		""" Flatten the DM by setting all segment coefficients to zero."""
		self._surface = None
		self._coef[:] = 0

	def set_segment(self, segid, piston, tip, tilt):
		""" Set an individual segment of the DM.

		Piston in meter of surface, tip and tilt in radians of surface.

		Parameters
		-------------
		segid : integer
			Index of the segment you wish to control, starting at 1 (center would  be 0, but doesn't exist)
		piston, tip, tilt : floats, meters and radians
			Piston (in meters) and tip and tilt (in radians)
		"""
		self._surface = None
		self._coef[segid - 1] = [piston, tip, tilt]

	def _setup_grids(self):
		""" Set up the grids to compute the segmented mirror surface into.
		This is relatively slow, but we only need to do this once for
		each size of input grids.
		"""
		npix = self.aperture.shaped.shape[0]
		if npix == self._last_npix:
			return
		else:
			self._last_npix = npix

		x, y = self.input_grid.coords

		self._seg_mask = np.zeros_like(x)
		self._seg_x = np.zeros_like(x)
		self._seg_y = np.zeros_like(y)
		self._seg_indices = dict()

		pupil_grid = make_pupil_grid(dims=npix, diameter=PUP_DIAMETER)
		aper_num, seg_positions = get_atlast_aperture(normalized=False, segment_transmissions=np.arange(1, self.segnum + 1), return_segment_positions=True)
		aper_num = aper_num(pupil_grid)

		self._seg_mask = np.copy(aper_num)

		for i in self.segmentlist:
			wseg = np.where(self._seg_mask == i)
			self._seg_indices[i] = wseg

			cenx, ceny = self.seg_pos.points[i - 1]

			self._seg_x[wseg] = x[wseg] - cenx
			self._seg_y[wseg] = y[wseg] - ceny

			# Set gaps to zero
			bad_gaps_x = np.where(
				np.abs(self._seg_x) > 0.1 * PUP_DIAMETER)  # *PUP_DIAMETER generalizes it for any size pupil field
			self._seg_x[bad_gaps_x] = 0
			bad_gaps_y = np.where(np.abs(self._seg_y) > 0.1 * PUP_DIAMETER)
			self._seg_y[bad_gaps_y] = 0

	def apply_coef(self):
		""" Apply the DM shape from its own segment coefficients to make segmented mirror surface.
		"""
		self._setup_grids()

		keep_surf = np.zeros_like(self._seg_x)
		for i in self.segmentlist:
			wseg = self._seg_indices[i]
			keep_surf[wseg] = (self._coef[i - 1, 0] +
							   self._coef[i - 1, 1] * self._seg_x[wseg] +
							   self._coef[i - 1, 2] * self._seg_y[wseg])
		return Field(keep_surf, self.input_grid)

	def phase_for(self, wavelength):
		"""Get the phase that is added to a wavefront with a specified wavelength.

		Parameters
		----------
		wavelength : scalar
			The wavelength at which to calculate the phase deformation.

		Returns
		-------
		Field
			The calculated phase deformation.
		"""
		return 2 * self.surface * 2 * np.pi / wavelength
