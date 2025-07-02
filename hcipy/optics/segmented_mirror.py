import numpy as np
import scipy.sparse

from ..mode_basis import ModeBasis, make_hexike_basis
from .deformable_mirror import DeformableMirror

class SegmentedDeformableMirror(DeformableMirror):
    '''A segmented deformable mirror with piston, tip, tilt, and optional Hexike control.

    This deformable mirror class can simulate devices such as those
    made by IrisAO and BMC. All segments are controlled in piston,
    tip and tilt by default. Higher-order aberrations can be
    enabled for each segment using hexagonal Hexike modes via a
    phase screen approach.

    Parameters
    ----------
    segments : ModeBasis
        A mode basis containing all segment apertures.
    segment_diameter : float, optional
        Physical diameter of segments in meters. Represents point-to-point length. If None, estimated from segment apertures.
    hexagon_angle : float, optional
        Rotation angle of hexagons in radians. Default is 0.
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
    
    where N is the number of segments.

    Examples
    --------
    >>> # PTT-only mirror (backward compatible)
    >>> mirror = SegmentedDeformableMirror(segments)
    >>> mirror.set_segment_actuators(0, 100e-9, 1e-6, 1e-6)
    
    >>> # PTT + Hexike mirror (phase screen approach)
    >>> mirror = SegmentedDeformableMirror(segments,
    ...                                   segment_centers=centers, pupil_grid=grid)
    >>> hexike_coeffs = {0: {4: 100}}  # segment 0, mode 4, 100nm amplitude
    >>> phase_screen = mirror.apply_segment_hexike_aberrations(hexike_coeffs, wavelength)
    '''
    def __init__(self, segments, segment_diameter=None, hexagon_angle=0,
                 segment_centers=None, pupil_grid=None):
        # Store configuration
        self.segment_diameter = segment_diameter
        self.hexagon_angle = hexagon_angle
        
        # Store geometry for hexike phase screen approach
        self.segment_centers = segment_centers
        self.pupil_grid = pupil_grid
        self.segment_point_to_point = segment_diameter
        
        # Initialize hexike phase screen system
        self._hexike_phase_screen = None
        self._hexike_coefficients = {}  # Store hexike coefficients: {segment_id: {mode: amplitude}}
        
        # Initialize actuators for PTT only
        self.actuators = np.zeros(len(segments) * 3)
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
        x = self.input_grid.x[segment_mask]
        y = self.input_grid.y[segment_mask]
        
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

        # Combine segments, tip, and tilt modes
        self.influence_functions = segments + tip + tilt

    def get_segment_actuators(self, segment_id):
        '''Get all actuators for an individual segment of the DM.

        Parameters
        ----------
        segment_id : int
            The index of the segment for which to get the actuators.

        Returns
        -------
        tuple
            (piston, tip, tilt) - Units: piston in meters, tip/tilt in radians.
        '''
        num_segments = len(self._segments)
        
        # PTT actuators only
        piston = self.actuators[segment_id]
        tip = self.actuators[segment_id + num_segments]  
        tilt = self.actuators[segment_id + 2 * num_segments]
        
        return (piston, tip, tilt)

    def set_segment_actuators(self, segment_id, piston, tip, tilt):
        '''Set actuators for an individual segment of the DM.

        Parameters
        ----------  
        segment_id : int
            The index of the segment for which to set the actuators.
        piston : float
            Piston actuator value in meters.
        tip : float
            Tip actuator value in radians.
        tilt : float
            Tilt actuator value in radians.
        '''
        num_segments = len(self._segments)
        
        # Set PTT actuators
        self.actuators[segment_id] = piston
        self.actuators[segment_id + num_segments] = tip
        self.actuators[segment_id + 2 * num_segments] = tilt


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
