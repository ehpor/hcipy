import numpy as np
import scipy.sparse

from ..field import make_hexagonal_grid
from ..mode_basis import ModeBasis, ansi_to_zernike, make_hexike_basis, zernike_to_noll
from .optical_element import OpticalElement
from ..aperture import make_hexagonal_segmented_aperture


class SegmentedHexikeSurface(OpticalElement):
    '''An optical element applying per-segment hexike surface aberrations.

    The coefficients represent surface height in meters. Phase is applied as
    ``phase = 4 * pi * surface / wavelength`` (reflection: OPD = 2 * surface).

    Parameters
    ----------
    segments : iterable of Field or callable
        Segment masks, one per segment. Callables will be evaluated on `pupil_grid`.
    segment_centers : Grid
        Center positions for each segment, same ordering as `segments`.
    segment_circum_diameter : scalar
        Circumscribed diameter of each hexagonal segment.
    pupil_grid : Grid
        Grid on which the wavefront is defined.
    num_modes : int
        Number of hexike modes per segment (Noll ordered).
    hexagon_angle : float
        Rotation of each hexagon. Default pi/2 for flat-top orientation.
    '''
    def __init__(self, segments, segment_centers, segment_circum_diameter, pupil_grid, num_modes, hexagon_angle=np.pi / 2):
        self.input_grid = pupil_grid
        self._num_segments = len(segments)
        self._num_modes = num_modes

        segment_masks = [seg(pupil_grid) if callable(seg) else seg for seg in segments]

        modes = []

        for mask, center in zip(segment_masks, segment_centers.points):
            local_grid = pupil_grid.shifted(-center)
            local_basis = make_hexike_basis(local_grid, num_modes, segment_circum_diameter, hexagon_angle=hexagon_angle)
            local_matrix = local_basis.transformation_matrix
            mask_arr = np.asarray(mask)

            for i in range(num_modes):
                mode = local_matrix[:, i] * mask_arr
                mode = scipy.sparse.csr_matrix(mode)
                mode.eliminate_zeros()
                modes.append(mode)

        if modes:
            self._basis = ModeBasis(modes, pupil_grid)
        else:
            self._basis = ModeBasis(np.zeros((pupil_grid.size, 0)), pupil_grid)

        self._coefficients = np.zeros((self._num_segments, self._num_modes))
        self._coefficients_for_cached_surface = None
        self._surface = self.input_grid.zeros()

    @property
    def coefficients(self):
        '''Surface height coefficients in meters, shape (num_segments, num_modes).'''
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        arr = np.asarray(value)
        if arr.ndim == 1:
            if arr.size != self._num_segments * self._num_modes:
                raise ValueError('Coefficient array has wrong size.')
            arr = arr.reshape(self._num_segments, self._num_modes)
        elif arr.shape != (self._num_segments, self._num_modes):
            raise ValueError('Coefficient array has wrong shape.')

        self._coefficients = arr

    def set_segment_coefficients(self, segment_id, coeffs_dict, indexing='noll'):
        '''Set coefficients for a single segment.

        Parameters
        ----------
        segment_id : int
            Index of the segment.
        coeffs_dict : dict
            Mapping from mode index to surface height in meters.
        indexing : {'noll', 'ansi'}
            Indexing scheme for supplied mode indices.
        '''
        for mode_idx, height in coeffs_dict.items():
            internal_idx = self._to_internal_mode_index(mode_idx, indexing)
            self._coefficients[segment_id, internal_idx] = height

    def set_coefficients_from_dict(self, coeffs_by_segment, indexing='noll'):
        '''Convenience setter for dict-of-dicts coefficient input.'''
        for seg_id, mode_dict in coeffs_by_segment.items():
            self.set_segment_coefficients(seg_id, mode_dict, indexing=indexing)

    def get_segment_coefficients(self, segment_id):
        '''Get coefficients for a single segment.'''
        return self._coefficients[segment_id].copy()

    def flatten(self):
        '''Reset all coefficients to zero.'''
        self._coefficients[:] = 0

    @property
    def surface(self):
        '''Current surface height in meters as a Field on `input_grid`.'''
        coeffs = self._coefficients.ravel()

        if self._coefficients_for_cached_surface is not None:
            if np.all(coeffs == self._coefficients_for_cached_surface):
                return self._surface

        self._surface = self._basis.linear_combination(coeffs)
        self._coefficients_for_cached_surface = coeffs.copy()

        return self._surface

    @property
    def opd(self):
        '''Optical path difference (2 × surface for reflection).'''
        return 2 * self.surface

    def phase_for(self, wavelength):
        '''Phase screen for a given wavelength.'''
        return 4 * np.pi * self.surface / wavelength

    def forward(self, wavefront):
        if not np.any(self._coefficients != 0):
            return wavefront.copy()

        phase = 4 * np.pi * self.surface / wavefront.wavelength

        wf = wavefront.copy()
        wf.electric_field *= np.exp(1j * phase)
        return wf

    def backward(self, wavefront):
        if not np.any(self._coefficients != 0):
            return wavefront.copy()

        phase = 4 * np.pi * self.surface / wavefront.wavelength

        wf = wavefront.copy()
        wf.electric_field *= np.exp(-1j * phase)
        return wf

    def _to_internal_mode_index(self, mode_index, indexing):
        if indexing == 'noll':
            internal_idx = mode_index - 1
        elif indexing == 'ansi':
            n, m = ansi_to_zernike(mode_index)
            internal_idx = zernike_to_noll(n, m) - 1
        else:
            raise ValueError('Indexing must be "noll" or "ansi".')

        if internal_idx < 0 or internal_idx >= self._num_modes:
            raise ValueError('Mode index out of range for this surface.')
        return internal_idx


def make_segment_hexike_surface_from_hex_aperture(num_rings, segment_flat_to_flat, gap_size, pupil_grid, num_modes, hexagon_angle=np.pi / 2, starting_ring=0):
    '''Factory for SegmentedHexikeSurface on a hexagonal segmented aperture.'''
    _, segment_generators = make_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size, starting_ring=starting_ring, return_segments=True)

    segments = [seg(pupil_grid) for seg in segment_generators]

    segment_pitch = segment_flat_to_flat + gap_size
    segment_centers = make_hexagonal_grid(segment_pitch, num_rings, pointy_top=False)

    if starting_ring != 0:
        starting_segment = 3 * (starting_ring - 1) * starting_ring + 1
        mask = segment_centers.zeros(dtype='bool')
        mask[starting_segment:] = True
        segment_centers = segment_centers.subset(mask)

    segment_circum_diameter = segment_flat_to_flat * 2 / np.sqrt(3)

    return SegmentedHexikeSurface(segments, segment_centers, segment_circum_diameter, pupil_grid, num_modes, hexagon_angle=hexagon_angle)
