import numpy as np
from scipy.signal import windows

from ..optics import OpticalElement, Apodizer, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid
from ..fourier import FastFourierTransform, MatrixFourierTransform, FourierFilter

class MultiScaleCoronagraph(OpticalElement):
    '''A phase mask coronagraph.

    This :class:`OpticalElement` simulates the propagation of light through
    a phase mask in the focal plane. To resolve the singularity of the
    phase plate, a multi-scale approach is made. Discretisation errors made at
    a certain level are corrected by the next level with finer sampling.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    complex_mask : field_generator
        The complex focal-plane phase mask.
    lyot_stop : Field or OpticalElement
        The Lyot stop for the coronagraph. If it's a Field, it is converted to an
        OpticalElement for convenience. If this is None (default), then no Lyot stop is used.
    q : scalar
        The minimum number of pixels per lambda/D. The number of levels in the multi-scale
        Fourier transforms will be chosen to reach at least this number of samples. The required
        q for a high-accuracy vortex coronagraph for example depends on the charge of the vortex. For charge 2,
        this can be as low as 32, but for charge 8 you need ~1024. Lower values give higher performance
        as a smaller number of levels is needed, but increases the sampling errors near the singularity.
        Charges not divisible by four require a much lower q. The default (q=1024) is conservative in
        most cases.
    scaling_factor : scalar
        The fractional increase in spatial frequency sampling per level. Larger scaling factors
        require a smaller number of levels, but each level requires a slower Fourier transform.
        Factors of 2 or 4 usually perform the best.
    window_size : integer
        The size of the next level in the number of pixels on the current layer. Lowering this
        increases performance in exchange for accuracy. Values smaller than 4-8 are not recommended.
    '''
    def __init__(self, input_grid, complex_mask, lyot_stop=None, q=1024, scaling_factor=4, window_size=32):
        self.input_grid = input_grid
        pupil_diameter = input_grid.shape * input_grid.delta

        if hasattr(lyot_stop, 'forward') or lyot_stop is None:
            self.lyot_stop = lyot_stop
        else:
            self.lyot_stop = Apodizer(lyot_stop)

        levels = int(np.ceil(np.log(q / 2) / np.log(scaling_factor))) + 1
        qs = [2 * scaling_factor**i for i in range(levels)]
        num_airys = [input_grid.shape / 2]

        focal_grids = []
        self.focal_masks = []
        self.props = []

        for i in range(1, levels):
            num_airys.append(num_airys[i - 1] * window_size / (2 * qs[i - 1] * num_airys[i - 1]))

        for i in range(levels):
            q = qs[i]
            num_airy = num_airys[i]

            focal_grid = make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=1, focal_length=1)
            focal_mask = complex_mask(focal_grid)

            if i != levels - 1:
                wx = windows.tukey(window_size, 1, False)
                wy = windows.tukey(window_size, 1, False)
                w = np.outer(wy, wx)

                w = np.pad(w, (focal_grid.shape - w.shape) // 2, 'constant').ravel()
                focal_mask *= 1 - w

            for j in range(i):
                fft = FastFourierTransform(focal_grids[j])
                mft = MatrixFourierTransform(focal_grid, fft.output_grid)

                focal_mask -= mft.backward(fft.forward(self.focal_masks[j]))

            if i == 0:
                prop = FourierFilter(input_grid, focal_mask, q)
            else:
                prop = FraunhoferPropagator(input_grid, focal_grid)

            focal_grids.append(focal_grid)
            self.focal_masks.append(focal_mask)
            self.props.append(prop)

    def forward(self, wavefront):
        '''Propagate a wavefront through the coronagraph.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is expected to be
            in the pupil plane.

        Returns
        -------
        Wavefront
            The Lyot plane wavefront.
        '''
        wavelength = wavefront.wavelength
        wavefront.wavelength = 1

        for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
            if i == 0:
                lyot = Wavefront(prop.forward(wavefront.electric_field), input_stokes_vector=wavefront.input_stokes_vector)
            else:
                focal = prop(wavefront)
                focal.electric_field *= mask
                lyot.electric_field += prop.backward(focal).electric_field

        lyot.wavelength = wavelength
        wavefront.wavelength = wavelength

        if self.lyot_stop is not None:
            lyot = self.lyot_stop.forward(lyot)

        return lyot

    def backward(self, wavefront):
        '''Propagate backwards through the coronagraph.

        This is essentially a forward propagation through the same
        coronagraph, but with the phase pattern inverted.

        Parameters
        ----------
        wavefront : Wavefront
            The Lyot plane wavefront.

        Returns
        -------
        Wavefront
            The pupil-plane wavefront.
        '''
        if self.lyot_stop is not None:
            wavefront = self.lyot_stop.backward(wavefront)

        wavelength = wavefront.wavelength
        wavefront.wavelength = 1

        for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
            if i == 0:
                pup = Wavefront(prop.backward(wavefront.electric_field), input_stokes_vector=wavefront.input_stokes_vector)
            else:
                focal = prop(wavefront)
                focal.electric_field *= mask.conj()
                pup.electric_field += prop.backward(focal).electric_field

        pup.wavelength = wavelength
        wavefront.wavelength = wavelength

        return pup
