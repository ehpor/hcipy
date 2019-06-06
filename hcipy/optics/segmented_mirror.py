import os
import numpy as np
import matplotlib.pyplot as plt

from .optical_element import OpticalElement
from ..field import Field
from ..plotting import imshow_field
from ..mode_basis import SparseModeBasis
from .deformable_mirror import DeformableMirror

class SegmentedDeformableMirror(DeformableMirror):
	def __init__(self, segments):
		self.segments = segments
		self.actuators = np.zeros(len(segments) * 3)
		self.input_grid = segments.grid
	
	@property
	def segments(self):
		return self._segments
	
	@segments.setter
	def segments(self, segments):
		self._segments = segments

		norm = (segments.transformation_matrix**2).mean(axis=0)

		tip = segments.transformation_matrix * segments.grid.x
		tip -= segments.transformation_matrix * (tip.mean(axis=0) / norm)
		tip = SparseModeBasis(tip)
		
		tilt = segments.transformation_matrix * segments.grid.y
		tilt -= segments.transformation_matrix * (tilt.mean(axis=0) / norm)
		tilt = SparseModeBasis(tilt)
		
		self.influence_functions = segments + tip + tilt
	
	def get_segment_actuators(self, segment_id):
		piston = self.actuators[segment_id]
		tip = self.actuators[segment_id + len(self._segments)]
		tilt = self.actuators[segment_id + 2 * len(self._segments)]

		return (piston, tip, tilt)

	def set_segment_actuators(self, segment_id, piston, tip, tilt):
		self.actuators[segment_id] = piston
		self.actuators[segment_id + len(self._segments)] = tip
		self.actuators[segment_id + 2 * len(self._segments)] = tilt
	
class SegmentedMirror(OpticalElement):
	"""A segmented mirror from a segmented aperture.

	Parameters:
	----------
	indexed_aperture : Field
		The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
		segment identification. Segment gaps must be strictly zero.
	seg_pos : CartesianGrid(UnstructuredCoords)
		Segment positions of the aperture.
	"""

	def __init__(self, indexed_aperture, seg_pos):
		self.ind_aper = indexed_aperture
		self.segnum = len(seg_pos.x)
		self.segmentlist = np.arange(1, self.segnum + 1)
		self._coef = np.zeros((self.segnum, 3))
		self.seg_pos = seg_pos
		self.input_grid = indexed_aperture.grid
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

	def show_numbers(self):
		""" Display the mirror pupil with numbered segments.
		"""
		imshow_field(self.ind_aper)
		for i, par in enumerate(self.seg_pos):
			plt.annotate(s=i+1, xy=par, xytext=par, color='white', fontweight='bold') #TODO: scale text size by segment size

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
		npix = self.ind_aper.shaped.shape[0]
		if npix == self._last_npix:
			return
		else:
			self._last_npix = npix

		x, y = self.input_grid.coords

		self._seg_x = np.zeros_like(x)
		self._seg_y = np.zeros_like(y)
		self._seg_indices = dict()

		for i in self.segmentlist:
			wseg = np.where(self.ind_aper == i)
			self._seg_indices[i] = wseg

			cenx, ceny = self.seg_pos.points[i - 1]

			self._seg_x[wseg] = x[wseg] - cenx
			self._seg_y[wseg] = y[wseg] - ceny

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
