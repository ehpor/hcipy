from .field import imshow_field
from ..field import Field
from ..config import Configuration

import numpy as np

def imshow_psf(
        psf, grid=None, vmin=1e-8, vmax=1e-1, scale='log',
        cmap=None, title=None, normalization='none',
        crosshairs=False, mark_centroid=False, colorbar=True,
        colorbar_orientation='vertical', spatial_resolution=1,
        ax=None, ticks=None, **kwargs):
    '''Display a PSF in a nice, consistent format.

    Parameters
    ----------
    psf : Field or Wavefront
        The point spread function that we want to display.
    grid : Grid or None
        If we want to redefine the grid of the psf.
    vmin : scalar
        The minimum value on the colorbar.
    vmax : scalar
        The maximum value on the colorbar.
    scale : ['linear', 'log', 'logarithmic']
        The scale of the image.
    cmap : matplotlib.cm.Colormap instance
        The colormap to use. If default, it will be taken from the configuration file.
    title : string, optional
        Give the image a title. If this is not given, no title will be displayed.
    normalization : ['peak', 'total', 'none'] or scalar
        The type of normalization to apply. 'peak' normalizes by the maximum intensity.
        'total' normalizes by the total intensity, 'none' doesn't do any normalization.
        If this is a scalar, the PSF will be divided by this number before displaying it.
    crosshairs : boolean
        Whether to draw crosshairs at the image center. Default: False.
    mark_centroid : boolean
        Whether to draw a small crosshair at the centroid of the PSF. The centroid will be
        computed with TODO Default: False.
    colorbar : boolean
        Whether to draw a colorbar on the image. Default: True.
    colorbar_orientation : ['horizontal', 'vertical']
        How the colorbar should be oriented.
    spatial_resolution : scalar
        This will be the size of a unit cell on the axes.
    ax : matplotlib axes
        The axes which to draw on. If it is not given, the current axes will be used.
    ticks : array_like
        The ticks on the added colorbar.
    kwargs : dictionary
        Any keyword arguments will be sent to `imshow_field()`.

    Returns
    -------
    Image
        The created image.
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if hasattr(psf, 'wavelength'):
        psf = psf.power

    if grid is not None:
        psf = Field(psf, grid)

    if normalization.lower() == 'peak':
        psf_norm = psf.max()
    elif normalization.lower() == 'total':
        psf_norm = np.sum(psf * psf.grid.weights)
    elif normalization.lower() == 'none':
        psf_norm = 1
    else:
        psf_norm = normalization

    if cmap is None:
        cmap = Configuration().plotting.psf_colormap

    img = psf / psf_norm

    if scale.lower() == 'linear':
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif scale.lower() == 'log' or scale.lower() == 'logarithmic':
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError('Scale needs to be one of ["linear", "log", "logarithmic"].')

    im = imshow_field(
        img, norm=norm, cmap=cmap,
        grid_units=spatial_resolution, ax=ax, **kwargs
    )

    if crosshairs:
        ax.axhline(0, ls=':', color='k')
        ax.axvline(0, ls=':', color='k')

    if title:
        ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation=colorbar_orientation)

        if scale.lower() in ['log', 'logarithmic']:
            if ticks is None:
                ticks = np.logspace(np.log10(vmin), np.log10(vmax), int(np.round(np.log10(vmax / vmin) + 1)))

                if colorbar_orientation == 'horizontal' and vmax == 1e-1 and vmin == 1e-8:
                    # Use better looking ticks
                    ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

            cb.set_ticks(ticks)
            cb.set_ticklabels(ticks)

        if normalization.lower() == 'peak':
            cb.set_label('Intensity relative to peak')
        elif normalization.lower() == 'total':
            cb.set_label('Fractional intensity per pixel')

    if mark_centroid:
        raise NotImplementedError()

    return im

def imshow_pupil_phase(
        pupil_phase, grid=None, phase_limits=None, vmin=None, vmax=None, cmap=None,
        colorbar=True, colorbar_orientation='vertical', title=None, crosshairs=False,
        remove_piston=False, ax=None, **kwargs):
    '''Display a pupil phase pattern in a nice, consistent format.

    Parameters
    ----------
    pupil_phase : Field or Wavefront
        The pupil that we want to display. If it is a wavefront or complex, the phase will be used
        as the phase, and the amplitude will be applied as mask.
    grid : Grid or None
        If we want to redefine the grid of the pupil.
    phase_limits : scalar
        The minimum and maximum of the pupil phase. If it is not given, and vmin and vmax are
        not given either, it will be estimated from the data. If the range is sufficiently close
        to 2 pi, the range will be extended to 2 pi. The vmin and vmax, unless overridden, will
        be set to -phase_limits and phase_limits, so the phase will be centered around zero.
    vmin : scalar, optional
        The minimum of the colorbar.
    vmax : scalar, optional
        The maximum of the colorbar.
    cmap : matplotlib.cm.Colormap instance
        The colormap to use. If default, it will be taken from the configuration file.
    colorbar : boolean
        Whether to draw a colorbar on the image. Default: True.
    colorbar_orientation : ['horizontal', 'vertical']
        How the colorbar should be oriented.
    title : string, optional
        Give the image a title. If this is not given, no title will be displayed.
    crosshairs : boolean
        Whether to draw crosshairs at the image center. Default: False.
    remove_piston : boolean
        Whether to remove piston from the phase. Default: False.
    ax : matplotlib axes
        The axes which to draw on. If it is not given, the current axes will be used.
    kwargs : dictionary
        Any keyword arguments will be sent to `imshow_field()`.
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if hasattr(pupil_phase, 'wavelength'):
        pupil_phase = pupil_phase.electric_field

    if grid is not None:
        pupil_phase = Field(pupil_phase, grid)

    if np.iscomplexobj(pupil_phase):
        phase = Field(np.angle(pupil_phase), pupil_phase.grid)
        mask = np.abs(pupil_phase)
    else:
        phase = pupil_phase
        mask = None

    if remove_piston:
        complex_field = np.exp(1j * phase)
        if mask is not None:
            complex_field *= mask

        complex_field /= np.mean(complex_field)
        phase = Field(np.angle(complex_field), phase.grid)

    if phase_limits is None:
        phase_limits = max(-phase[mask > 0].min(), phase[mask > 0].max())

        if np.abs(phase_limits - np.pi) < 0.5:
            phase_limits = np.pi

        if phase_limits < 1e-10:
            phase_limits = 1e-10

    if vmin is None and vmax is None:
        vmin = -phase_limits
        vmax = phase_limits

    if cmap is None:
        cmap = Configuration().plotting.pupil_phase_colormap

    im = imshow_field(phase, ax=ax, cmap=cmap, mask=mask, vmin=vmin, vmax=vmax, **kwargs)

    if crosshairs:
        ax.axhline(0, ls=':', color='k')
        ax.axvline(0, ls=':', color='k')

    if title:
        ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation=colorbar_orientation)
        cb.set_label('Phase [rad]')

    return im
