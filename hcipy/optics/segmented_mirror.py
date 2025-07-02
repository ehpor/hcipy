import numpy as np
import scipy.sparse

from ..mode_basis import ModeBasis, make_zernike_basis, noll_to_zernike, ansi_to_zernike, zernike_to_noll, zernike_to_ansi
from .deformable_mirror import DeformableMirror

class SegmentedDeformableMirror(DeformableMirror):
    '''A segmented deformable mirror with piston, tip, tilt, and optional Zernike control.

    This deformable mirror class can simulate devices such as those
    made by IrisAO and BMC. All segments are controlled in piston,
    tip and tilt by default. Higher-order Zernike aberrations can be
    optionally enabled for each segment.

    Parameters
    ----------
    segments : ModeBasis
        A mode basis containing all segment apertures.
    num_zernike_modes : int, optional
        Number of Zernike modes per segment beyond piston/tip/tilt. Default is 0.
    zernike_starting_mode : int, optional
        Starting Zernike mode index. Default is 1 (includes all Zernike modes).
    zernike_indexing : str, optional
        Indexing scheme: 'noll' or 'ansi'. Default is 'noll'.
    segment_diameter : float, optional
        Physical diameter of segments in meters. If None, estimated from segment apertures.

    Notes
    -----
    Actuator organization:
    - segment_id + 0*N: Piston for segment segment_id
    - segment_id + 1*N: Tip for segment segment_id  
    - segment_id + 2*N: Tilt for segment segment_id
    - segment_id + (2+i)*N: Zernike mode i for segment segment_id (if enabled)
    
    where N is the number of segments.

    Examples
    --------
    >>> # PTT-only mirror (backward compatible)
    >>> mirror = SegmentedDeformableMirror(segments)
    >>> mirror.set_segment_actuators(0, 100e-9, 1e-6, 1e-6)
    
    >>> # PTT + Zernike mirror
    >>> mirror = SegmentedDeformableMirror(segments, num_zernike_modes=10)
    >>> mirror.set_segment_zernike_actuator(segment_id=0, zernike_index=5, amplitude=50e-9)
    '''
    def __init__(self, segments, num_zernike_modes=0, zernike_starting_mode=1, 
                 zernike_indexing='noll', segment_diameter=None):
        # Store Zernike configuration
        self.num_zernike_modes = num_zernike_modes
        self.zernike_starting_mode = zernike_starting_mode
        self.zernike_indexing = zernike_indexing
        self.segment_diameter = segment_diameter
        
        # Validate parameters
        if num_zernike_modes < 0:
            raise ValueError("num_zernike_modes must be non-negative")
        if zernike_indexing not in ['noll', 'ansi']:
            raise ValueError("zernike_indexing must be 'noll' or 'ansi'")
        if zernike_starting_mode < 1:
            raise ValueError("zernike_starting_mode must be >= 1")
        
        # Initialize actuators for PTT + Zernike modes
        actuators_per_segment = 3 + num_zernike_modes
        self.actuators = np.zeros(len(segments) * actuators_per_segment)
        self.input_grid = segments.grid
        
        # Set segments (this will trigger influence function generation)
        self.segments = segments

    def _estimate_segment_diameter(self, segment_id):
        '''Estimate the diameter of a segment from its aperture mask.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
            
        Returns
        -------
        float
            Estimated diameter in the same units as the grid.
        '''
        segment = self.segments[segment_id]
        segment_mask = segment > 0.5
        
        if not np.any(segment_mask):
            raise ValueError(f"Segment {segment_id} appears to be empty")
        
        # Get coordinates where segment is non-zero
        if segment_mask.ndim == 1:
            # Field is 1D (flattened), need to convert to physical coordinates
            x = self.input_grid.x[segment_mask]
            y = self.input_grid.y[segment_mask]
        else:
            # Field is 2D, use np.where
            y_coords, x_coords = np.where(segment_mask)
            x = x_coords * abs(self.input_grid.delta[0]) + self.input_grid.zero[0]
            y = y_coords * abs(self.input_grid.delta[1]) + self.input_grid.zero[1]
        
        # Calculate extents
        x_extent = x.max() - x.min()
        y_extent = y.max() - y.min()
        
        return max(x_extent, y_extent)

    def _get_segment_center(self, segment_id):
        '''Calculate the centroid of a segment.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
            
        Returns
        -------
        ndarray
            Array containing [x_center, y_center] coordinates.
        '''
        segment = self.segments[segment_id]
        x = self.input_grid.x
        y = self.input_grid.y
        
        total_weight = segment.sum()
        if total_weight == 0:
            raise ValueError(f"Segment {segment_id} has zero weight")
        
        x_center = (segment * x).sum() / total_weight
        y_center = (segment * y).sum() / total_weight
        
        return np.array([x_center, y_center])

    def _orthogonalize_against_ptt(self, mode, segment_id):
        '''Orthogonalize a mode against piston, tip, and tilt for a specific segment.
        
        Note: This is called before PTT modes are added to influence_functions,
        so we need to generate them on-the-fly.
        
        Parameters
        ----------
        mode : Field
            The mode to orthogonalize.
        segment_id : int
            The index of the segment.
            
        Returns
        -------
        Field
            The orthogonalized mode.
        '''
        segment = self.segments[segment_id]
        
        # Generate PTT modes on-the-fly (since influence_functions aren't complete yet)
        segment_mean = segment.mean()
        norm = np.mean(segment**2) - segment_mean**2
        
        # Piston mode
        piston_mode = segment
        
        # Tip mode
        tip_mode = segment * segment.grid.x
        if norm != 0:
            beta = ((tip_mode * segment).mean() - segment_mean * tip_mode.mean()) / norm
            tip_mode -= beta * segment
            
        # Tilt mode  
        tilt_mode = segment * segment.grid.y
        if norm != 0:
            beta = ((tilt_mode * segment).mean() - segment_mean * tilt_mode.mean()) / norm
            tilt_mode -= beta * segment
        
        # Gram-Schmidt orthogonalization against PTT modes
        for existing_mode in [piston_mode, tip_mode, tilt_mode]:
            norm_existing = (existing_mode * existing_mode).sum()
            if norm_existing > 0:
                projection = (mode * existing_mode).sum() / norm_existing
                mode = mode - projection * existing_mode
        
        return mode

    def _generate_segment_zernike_modes(self):
        '''Generate Zernike modes for all segments.
        
        Returns
        -------
        ModeBasis
            A mode basis containing all segment-level Zernike modes.
        '''
        if self.num_zernike_modes == 0:
            return ModeBasis([])
        
        all_zernike_modes = []
        
        for segment_id in range(len(self.segments)):
            segment = self.segments[segment_id]
            segment_center = self._get_segment_center(segment_id)
            
            # Use provided diameter or estimate from segment
            if self.segment_diameter is not None:
                diameter = self.segment_diameter
            else:
                diameter = self._estimate_segment_diameter(segment_id)
            
            # Create segment-local grid centered on this segment
            segment_grid = self.input_grid.shifted(-segment_center)
            
            # Generate Zernike basis for this segment  
            zernike_basis = make_zernike_basis(
                self.num_zernike_modes,
                diameter,
                segment_grid,
                starting_mode=self.zernike_starting_mode,
                ansi=(self.zernike_indexing == 'ansi'),
                radial_cutoff=True
            )
            
            # Apply segment mask and orthogonalize against PTT
            for mode in zernike_basis:
                # Mask mode to segment
                masked_mode = mode * segment
                
                # Orthogonalize against existing PTT modes for this segment
                masked_mode = self._orthogonalize_against_ptt(masked_mode, segment_id)
                
                # Convert to sparse matrix and eliminate zeros
                sparse_mode = scipy.sparse.csr_matrix(masked_mode)
                sparse_mode.eliminate_zeros()
                
                all_zernike_modes.append(sparse_mode)
        
        return ModeBasis(all_zernike_modes)

    def _generate_ptt_modes(self):
        '''Generate piston, tip, and tilt modes (extracted from existing logic).
        
        Returns
        -------
        tuple
            (tip_modes, tilt_modes) as ModeBasis objects.
        '''
        tip = []
        tilt = []
        
        for segment in self._segments:
            segment_mean = segment.mean()
            norm = np.mean(segment**2) - segment_mean**2
            
            # Tip mode generation
            tip_mode = segment * segment.grid.x
            if norm != 0:
                beta = ((tip_mode * segment).mean() - segment_mean * tip_mode.mean()) / norm
                tip_mode -= beta * segment
            
            tip_mode = scipy.sparse.csr_matrix(tip_mode)
            tip_mode.eliminate_zeros()
            tip.append(tip_mode)
            
            # Tilt mode generation  
            tilt_mode = segment * segment.grid.y
            if norm != 0:
                beta = ((tilt_mode * segment).mean() - segment_mean * tilt_mode.mean()) / norm
                tilt_mode -= beta * segment
                
            tilt_mode = scipy.sparse.csr_matrix(tilt_mode)
            tilt_mode.eliminate_zeros()
            tilt.append(tilt_mode)
        
        return ModeBasis(tip), ModeBasis(tilt)

    @property
    def segments(self):
        '''The segments of this deformable mirror in a ModeBasis.
        '''
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

        # Generate PTT modes using extracted logic
        tip, tilt = self._generate_ptt_modes()

        # Combine segments, tip, tilt, and optionally Zernike modes
        self.influence_functions = segments + tip + tilt
        
        # Add Zernike modes if requested
        if self.num_zernike_modes > 0:
            zernike_modes = self._generate_segment_zernike_modes()
            self.influence_functions = self.influence_functions + zernike_modes

    def get_segment_actuators(self, segment_id):
        '''Get all actuators for an individual segment of the DM.

        Parameters
        ----------
        segment_id : int
            The index of the segment for which to get the actuators.

        Returns
        -------
        tuple
            If num_zernike_modes == 0: (piston, tip, tilt)
            If num_zernike_modes > 0: (piston, tip, tilt, z4, z5, z6, ...)
            Units: piston in meters, tip/tilt in radians, Zernike in meters RMS.
        '''
        actuators = []
        num_segments = len(self._segments)
        
        # PTT actuators
        piston = self.actuators[segment_id]
        tip = self.actuators[segment_id + num_segments]  
        tilt = self.actuators[segment_id + 2 * num_segments]
        actuators.extend([piston, tip, tilt])
        
        # Zernike actuators
        for zernike_idx in range(self.num_zernike_modes):
            actuator_offset = (3 + zernike_idx) * num_segments
            zernike_value = self.actuators[segment_id + actuator_offset]
            actuators.append(zernike_value)
        
        return tuple(actuators)

    def set_segment_actuators(self, segment_id, *actuator_values):
        '''Set actuators for an individual segment of the DM.

        Parameters
        ----------  
        segment_id : int
            The index of the segment for which to set the actuators.
        *actuator_values : float
            Actuator values. Must match expected number:
            - 3 values for PTT-only: (piston, tip, tilt)
            - 3+N values for PTT+Zernike: (piston, tip, tilt, z4, z5, ...)
            Units: piston in meters, tip/tilt in radians, Zernike in meters RMS.
        '''
        expected_actuators = 3 + self.num_zernike_modes
        
        if len(actuator_values) != expected_actuators:
            raise ValueError(f"Expected {expected_actuators} actuator values, got {len(actuator_values)}")
        
        num_segments = len(self._segments)
        
        # Set PTT actuators
        self.actuators[segment_id] = actuator_values[0]  # piston
        self.actuators[segment_id + num_segments] = actuator_values[1]  # tip
        self.actuators[segment_id + 2 * num_segments] = actuator_values[2]  # tilt
        
        # Set Zernike actuators
        for zernike_idx in range(self.num_zernike_modes):
            actuator_offset = (3 + zernike_idx) * num_segments
            self.actuators[segment_id + actuator_offset] = actuator_values[3 + zernike_idx]

    def set_segment_zernike_actuator(self, segment_id, zernike_index, amplitude):
        '''Set a specific Zernike actuator for a segment.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
        zernike_index : int  
            The index of the Zernike mode (0-based relative to starting mode).
        amplitude : float
            The amplitude in meters RMS.
        '''
        if zernike_index >= self.num_zernike_modes:
            raise ValueError(f"Zernike index {zernike_index} >= num_zernike_modes {self.num_zernike_modes}")
        
        actuator_offset = (3 + zernike_index) * len(self._segments)
        self.actuators[segment_id + actuator_offset] = amplitude

    def get_segment_zernike_actuator(self, segment_id, zernike_index):
        '''Get a specific Zernike actuator value for a segment.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
        zernike_index : int
            The index of the Zernike mode (0-based relative to starting mode).
            
        Returns
        -------
        float
            The amplitude in meters RMS.
        '''
        if zernike_index >= self.num_zernike_modes:
            raise ValueError(f"Zernike index {zernike_index} >= num_zernike_modes {self.num_zernike_modes}")
        
        actuator_offset = (3 + zernike_index) * len(self._segments)
        return self.actuators[segment_id + actuator_offset]

    @property
    def num_actuators_per_segment(self):
        '''The number of actuators per segment (3 + num_zernike_modes).'''
        return 3 + self.num_zernike_modes

    @property  
    def total_num_actuators(self):
        '''The total number of actuators.'''
        return len(self._segments) * self.num_actuators_per_segment

    def get_zernike_mode_info(self, zernike_index):
        '''Get information about a specific Zernike mode.
        
        Parameters
        ----------
        zernike_index : int
            The index of the Zernike mode (0-based relative to starting mode).
            
        Returns
        -------
        dict
            Dictionary containing 'noll_index', 'ansi_index', 'n', 'm' keys.
        '''
        if zernike_index >= self.num_zernike_modes:
            raise ValueError(f"Zernike index {zernike_index} >= num_zernike_modes {self.num_zernike_modes}")
        
        if self.zernike_indexing == 'noll':
            noll_index = self.zernike_starting_mode + zernike_index
            n, m = noll_to_zernike(noll_index)
            ansi_index = zernike_to_ansi(n, m)
        else:  # ansi
            ansi_index = self.zernike_starting_mode + zernike_index
            n, m = ansi_to_zernike(ansi_index)
            noll_index = zernike_to_noll(n, m)
        
        return {
            'noll_index': noll_index,
            'ansi_index': ansi_index,
            'n': n,
            'm': m
        }
