import numpy as np
import scipy.sparse

from ..mode_basis import ModeBasis, make_zernike_basis, make_hexike_basis, noll_to_zernike, ansi_to_zernike, zernike_to_noll, zernike_to_ansi
from .deformable_mirror import DeformableMirror

class SegmentedDeformableMirror(DeformableMirror):
    '''A segmented deformable mirror with piston, tip, tilt, and optional Zernike/Hexike control.

    This deformable mirror class can simulate devices such as those
    made by IrisAO and BMC. All segments are controlled in piston,
    tip and tilt by default. Higher-order aberrations can be
    optionally enabled for each segment using either circular Zernike 
    modes or hexagonal Hexike modes.

    Parameters
    ----------
    segments : ModeBasis
        A mode basis containing all segment apertures.
    num_zernike_modes : int, optional
        Number of modes per segment beyond piston/tip/tilt. Default is 0.
    zernike_starting_mode : int, optional
        Starting Zernike mode index. Default is 1 (includes all Zernike modes).
        Only used when segment_mode_type='circular'.
    zernike_indexing : str, optional
        Indexing scheme: 'noll' or 'ansi'. Default is 'noll'.
        Only used when segment_mode_type='circular'.
    segment_diameter : float, optional
        Physical diameter of segments in meters. Represents point-to-point length. If None, estimated from segment apertures.
    segment_mode_type : str, optional
        Type of modes to use: 'circular' for Zernike modes or 'hexagonal' for Hexike modes.
        Default is 'circular'.
    hexagon_angle : float, optional
        Rotation angle of hexagons in radians. Default is 0.
        Only used when segment_mode_type='hexagonal'.
    segment_centers : Grid, optional
        Grid containing center positions of each segment. Required for hexike phase screen approach.
    pupil_grid : Grid, optional
        The main pupil grid. Required for hexike phase screen approach.

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
    
    >>> # PTT + Hexike mirror (phase screen approach)
    >>> mirror = SegmentedDeformableMirror(segments, segment_mode_type='hexagonal',
    ...                                   segment_centers=centers, pupil_grid=grid)
    >>> hexike_coeffs = {0: {4: 100}}  # segment 0, mode 4, 100nm amplitude
    >>> phase_screen = mirror.apply_segment_hexike_aberrations(hexike_coeffs, wavelength)
    '''
    def __init__(self, segments, num_zernike_modes=0, zernike_starting_mode=1, 
                 zernike_indexing='noll', segment_diameter=None,
                 segment_mode_type='circular', hexagon_angle=0,
                 segment_centers=None, pupil_grid=None):
        # Store Zernike configuration
        self.num_zernike_modes = num_zernike_modes
        self.zernike_starting_mode = zernike_starting_mode
        self.zernike_indexing = zernike_indexing
        self.segment_diameter = segment_diameter
        self.segment_mode_type = segment_mode_type
        self.hexagon_angle = hexagon_angle
        
        # Store geometry for hexike phase screen approach
        self.segment_centers = segment_centers
        self.pupil_grid = pupil_grid
        self.segment_point_to_point = segment_diameter
        
        # Initialize hexike phase screen system
        self._hexike_phase_screen = None
        self._hexike_coefficients = {}  # Store hexike coefficients: {segment_id: {mode: amplitude}}
        
        # Validate parameters
        if num_zernike_modes < 0:
            raise ValueError("num_zernike_modes must be non-negative")
        if zernike_indexing not in ['noll', 'ansi']:
            raise ValueError("zernike_indexing must be 'noll' or 'ansi'")
        if zernike_starting_mode < 1:
            raise ValueError("zernike_starting_mode must be >= 1")
        if segment_mode_type not in ['circular', 'hexagonal']:
            raise ValueError("segment_mode_type must be 'circular' or 'hexagonal'")
        
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
            # Field is 1D, convert to physical coordinates
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
        
        # Generate PTT modes for orthogonalization
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
            
            # Generate basis for this segment (Zernike or Hexike)
            if self.segment_mode_type == 'hexagonal':
                basis = make_hexike_basis(
                    self.num_zernike_modes,
                    diameter,
                    segment_grid,
                    hexagon_angle=self.hexagon_angle
                )
            else:  # circular
                basis = make_zernike_basis(
                    self.num_zernike_modes,
                    diameter,
                    segment_grid,
                    starting_mode=self.zernike_starting_mode,
                    ansi=(self.zernike_indexing == 'ansi'),
                    radial_cutoff=True
                )
            
            # Apply segment mask and conditionally orthogonalize against PTT
            for mode in basis:
                # Mask mode to segment
                masked_mode = mode * segment
                
                # Apply RMS normalization for hexagonal modes
                if self.segment_mode_type == 'hexagonal':
                    segment_mask = segment > 0.5
                    if np.any(segment_mask):
                        # Calculate RMS over the segment
                        mode_values = masked_mode[segment_mask]
                        rms = np.sqrt(np.mean(mode_values**2))
                        if rms > 0:
                            masked_mode = masked_mode / rms

                # Convert to sparse matrix and eliminate zeros
                sparse_mode = scipy.sparse.csr_matrix(masked_mode)
                sparse_mode.eliminate_zeros()
                
                all_zernike_modes.append(sparse_mode)
        
        return ModeBasis(all_zernike_modes)

    def _generate_ptt_modes(self):
        '''Generate piston, tip, and tilt modes.

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

        # Generate PTT modes
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

    def apply_segment_hexike_aberrations(self, segment_zernike_dict, wavelength):
        '''Apply hexike aberrations to individual segments using phase screen approach.

        Generates hexike modes on-demand for each segment and creates a phase screen.

        Parameters
        ----------
        segment_zernike_dict : dict
            Dictionary mapping segment ID to another dict of {mode: amplitude_nm}.
            Example: {0: {4: 20, 5: 10}, 5: {6: 15}} applies mode 4=20nm and mode 5=10nm
            to segment 0, and mode 6=15nm to segment 5.
        wavelength : float
            Wavelength in meters.

        Returns
        -------
        phase_screen : Field
            Phase screen containing segment-level hexike aberrations.

        Raises
        ------
        ValueError
            If required geometry parameters are not provided.
        '''
        if self.segment_centers is None or self.pupil_grid is None:
            raise ValueError("segment_centers and pupil_grid must be provided for hexike aberrations")
        
        if self.segment_point_to_point is None:
            raise ValueError("segment_diameter must be provided for hexike aberrations")

        # Store coefficients for automatic application
        self._hexike_coefficients = segment_zernike_dict.copy()

        # Initialize phase screen
        phase_screen = self.pupil_grid.zeros()

        # Process each segment that has aberrations
        for seg_id, mode_dict in segment_zernike_dict.items():

            if seg_id >= len(self.segments):
                # Warn for non-existent segments
                import warnings
                warnings.warn(f"Segment {seg_id} does not exist. "
                            f"Mirror has {len(self.segments)} segments. Skipping.",
                            UserWarning)
                continue
            
            if seg_id >= len(self.segment_centers.points):
                # Warn for segments without center information
                import warnings
                warnings.warn(f"Segment {seg_id} has no center information in segment_centers. "
                            f"Available centers: {len(self.segment_centers.points)}, "
                            f"requested segment: {seg_id}. Skipping this segment.",
                            UserWarning)
                continue
            
            # Get the center of this segment
            center = self.segment_centers.points[seg_id]

            # Find the maximum mode needed for this segment
            max_mode_for_segment = max(mode_dict.keys())

            # Create hexike basis for this segment
            angle = self.hexagon_angle
            basis = make_hexike_basis(int(max_mode_for_segment + 1), 
                                    self.segment_point_to_point,
                                    self.pupil_grid.shifted(-center), angle)

            # Apply each requested mode
            for mode, coeff_nm in mode_dict.items():
                if mode < len(basis):
                    # Convert nm to phase in radians
                    phase_rad = 2 * np.pi * (coeff_nm * 1e-9) / wavelength
                    # Get the mode as a Field
                    mode_field = basis[mode]
                    # Apply segment mask to ensure mode only affects this segment
                    segment_mask = self.segments[seg_id]
                    phase_screen += phase_rad * mode_field * segment_mask

        # Store the phase screen for automatic application
        self._hexike_phase_screen = phase_screen
        
        return phase_screen

    def __call__(self, wavefront):
        '''Apply the segmented deformable mirror to a wavefront.
        
        This method applies both the PTT actuator effects (via parent class)
        and any hexike phase screen aberrations.
        
        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to which to apply the mirror.
            
        Returns
        -------
        Wavefront
            The wavefront after applying the mirror effects.
        '''
        # Apply PTT deformable mirror effects using parent class
        wf = super().__call__(wavefront)
        
        # Apply hexike phase screen if it exists
        if self._hexike_phase_screen is not None:
            wf.electric_field *= np.exp(1j * self._hexike_phase_screen)
        
        return wf

    def set_segment_hexike_coefficient(self, segment_id, mode, amplitude_nm, wavelength):
        '''Set a hexike coefficient for a specific segment and mode.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
        mode : int
            The hexike mode index (0-based).
        amplitude_nm : float
            The amplitude in nanometers RMS.
        wavelength : float
            Wavelength in meters for phase conversion.
        '''
        if segment_id not in self._hexike_coefficients:
            self._hexike_coefficients[segment_id] = {}
        
        self._hexike_coefficients[segment_id][mode] = amplitude_nm
        
        # Regenerate phase screen with updated coefficients
        if self._hexike_coefficients:
            self.apply_segment_hexike_aberrations(self._hexike_coefficients, wavelength)

    def get_segment_hexike_coefficient(self, segment_id, mode):
        '''Get a hexike coefficient for a specific segment and mode.
        
        Parameters
        ----------
        segment_id : int
            The index of the segment.
        mode : int
            The hexike mode index (0-based).
            
        Returns
        -------
        float
            The amplitude in nanometers RMS, or 0 if not set.
        '''
        if segment_id in self._hexike_coefficients:
            return self._hexike_coefficients[segment_id].get(mode, 0.0)
        return 0.0

    def clear_hexike_aberrations(self):
        '''Clear all hexike aberrations and reset phase screen.'''
        self._hexike_coefficients = {}
        self._hexike_phase_screen = None

    def get_hexike_coefficients(self):
        '''Get all current hexike coefficients.
        
        Returns
        -------
        dict
            Dictionary mapping segment_id to {mode: amplitude_nm} dicts.
        '''
        return self._hexike_coefficients.copy()

    def update_hexike_phase_screen(self, wavelength):
        '''Regenerate the hexike phase screen from stored coefficients.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in meters for phase conversion.
            
        Returns
        -------
        Field or None
            The updated phase screen, or None if no coefficients are set.
        '''
        if self._hexike_coefficients:
            return self.apply_segment_hexike_aberrations(self._hexike_coefficients, wavelength)
        else:
            self._hexike_phase_screen = None
            return None
