import numpy as np

from ..optics import Apodizer, OpticalElement
from ..propagation import FraunhoferPropagator

class LyotCoronagraph(OpticalElement):
    '''A Lyot coronagraph with a small focal-plane mask.

    The area outside of this focal-plane mask is assumed to be fully transmisive. The
    method for propagation is based on [Soummer2007]_.

    .. [Soummer2007] Soummer et al. 2007, "Fast computation of Lyot-style
        coronagraph propagation".

    Notes
    -----
    The :class:`LyotCoronagraph` tries to automatically get the grid from the focal_plane_mask.
    However, a bug in AgnosticOpticalElements creates an invalid access which crashes the initialization.
    The correct grid can be passed explicitely to circumvent this bug by using the focal_plane_mask_grid parameter.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    focal_plane_mask : Field or OpticalElement
        The (complex) transmission of the focal-plane mask. If this is an :class:`OpticalElement`,
        this will be used instead. This allows for more realistic implementations of focal-plane
        masks.
    lyot_stop : Field or OpticalElement or None
        The (complex) transmission of the Lyot stop. If this is an :class:`OpticalElement`,
        this will be used instead. This allows for more realistic implementations of Lyot stops.
    focal_length : scalar
        The internal focal length of the Lyot system.
    focal_plane_mask_grid : Grid
        The grid on which the focal plane mask is defined. If this is none,
        the grid will be determined from the focal plane mask. The default value is None.
    '''
    def __init__(self, input_grid, focal_plane_mask, lyot_stop=None, focal_length=1, focal_plane_mask_grid=None):
        if hasattr(focal_plane_mask, 'input_grid'):
            # Focal plane mask is an optical element.
            if focal_plane_mask_grid is None:
                grid = focal_plane_mask.apodization.grid
            else:
                grid = focal_plane_mask_grid

            self.focal_plane_mask = focal_plane_mask
        else:
            # Focal plane mask is a field.
            if focal_plane_mask_grid is None:
                grid = focal_plane_mask.grid
            else:
                grid = focal_plane_mask_grid

            self.focal_plane_mask = Apodizer(focal_plane_mask)

        if lyot_stop is not None and not hasattr(lyot_stop, 'input_grid'):
            lyot_stop = Apodizer(lyot_stop)
        self.lyot_stop = lyot_stop

        self.prop = FraunhoferPropagator(input_grid, grid, focal_length=focal_length)

    def forward(self, wavefront):
        '''Propagate the wavefront through the Lyot coronagraph.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is assumed to be in the pupil plane.

        Returns
        -------
        Wavefront
            The Lyot-plane wavefront.
        '''
        wf_foc = self.prop.forward(wavefront)
        wf_foc.electric_field -= self.focal_plane_mask.forward(wf_foc).electric_field

        lyot = self.prop.backward(wf_foc)

        # The next line is equal to
        #     lyot.electric_field = wavefront.electric_field - lyot.electric_field
        # but doesn't copy a numpy array.
        np.subtract(wavefront.electric_field, lyot.electric_field, out=lyot.electric_field)

        if self.lyot_stop is not None:
            lyot = self.lyot_stop.forward(lyot)

        return lyot

    def backward(self, wavefront):
        '''Propagate the wavefront from the Lyot plane to the pupil plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        Wavefront
            The pupil-plane wavefront.
        '''
        if self.lyot_stop is not None:
            wf = self.lyot_stop.backward(wavefront)
        else:
            wf = wavefront

        wf_foc = self.prop.forward(wf)
        wf_foc.electric_field -= self.focal_plane_mask.backward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)

        # The next line is equal to
        #     pup.electric_field = wf.electric_field - pup.electric_field
        # but doesn't copy a numpy array.
        np.subtract(wf.electric_field, pup.electric_field, out=pup.electric_field)

        return pup

class OccultedLyotCoronagraph(OpticalElement):
    '''A Lyot coronagraph with a focal-plane mask.

    The area outside of this focal-plane mask is assumed to be fully absorbing.

    Notes
    -----
    The :class:`OccultedLyotCoronagraph` tries to automatically get the grid from the focal_plane_mask.
    However, a bug in AgnosticOpticalElements creates an invalid access which crashes the initialization.
    The correct grid can be passed explicitely to circumvent this bug by using the focal_plane_mask_grid parameter.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    focal_plane_mask : Field or OpticalElement
        The (complex) transmission of the focal-plane mask. If this is an :class:`OpticalElement`,
        this will be used instead. This allows for more realistic implementations of focal-plane
        masks.
    lyot_stop : Field or OpticalElement or None
        The (complex) transmission of the Lyot stop. If this is an :class:`OpticalElement`,
        this will be used instead. This allows for more realistic implementations of Lyot stops.
    focal_length : scalar
        The internal focal length of the Lyot system.
    focal_plane_mask_grid : Grid
        The grid on which the focal plane mask is defined. If this is none,
        the grid will be determined from the focal plane mask. The default value is None.
    '''
    def __init__(self, input_grid, focal_plane_mask, focal_length=1, focal_plane_mask_grid=None):

        if hasattr(focal_plane_mask, 'input_grid'):
            # Focal plane mask is an optical element.
            if focal_plane_mask_grid is None:
                grid = focal_plane_mask.apodization.grid
            else:
                grid = focal_plane_mask_grid

            self.focal_plane_mask = focal_plane_mask
        else:
            # Focal plane mask is a field.
            if focal_plane_mask_grid is None:
                grid = focal_plane_mask.grid
            else:
                grid = focal_plane_mask_grid

            self.focal_plane_mask = Apodizer(focal_plane_mask)

        self.prop = FraunhoferPropagator(input_grid, grid, focal_length=focal_length)

    def forward(self, wavefront):
        '''Propagate the wavefront through the Lyot coronagraph.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is assumed to be in the pupil plane.

        Returns
        -------
        Wavefront
            The Lyot-plane wavefront.
        '''
        return self.prop.backward(self.focal_plane_mask.forward(self.prop.forward(wavefront)))

    def backward(self, wavefront):
        '''Propagate the wavefront from the Lyot plane to the pupil plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        Wavefront
            The pupil-plane wavefront.
        '''
        return self.prop.backward(self.focal_plane_mask.backward(self.prop.forward(wavefront)))
