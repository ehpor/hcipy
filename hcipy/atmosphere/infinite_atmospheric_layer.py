from __future__ import division

from .atmospheric_model import AtmosphericLayer, phase_covariance_von_karman, fried_parameter_from_Cn_squared
from ..field import Field, RegularCoords, UnstructuredCoords, CartesianGrid
from .finite_atmospheric_layer import FiniteAtmosphericLayer

import numpy as np
from scipy import linalg
from scipy.ndimage import affine_transform

import warnings
import copy

class InfiniteAtmosphericLayer(AtmosphericLayer):
    '''An atmospheric layer that can be infinitely extended in any direction.

    This is an implementation of [Assemat2006]_. Contrary to the
    FiniteAtmosphericLayer class, this atmospheric layer uses an
    autoregressive model for extending the phase screen by one
    column or row in any direction. Each new row/column is based on
    a  stencil, which contains the last few rows/columns in addition
    to a few pixels spread out more into the phase screen to capture
    the low-order aberrations.

    .. [Assemat2006] François Assémat, Richard W. Wilson, and Eric
        Gendron, "Method for simulating infinitely long and non
        stationary phase screens with optimized memory storage,"
        Opt. Express 14, 988-999 (2006)

    .. note::
        This algorithm does not work well with large outer scales.
        This is an inherent flaw in the algorithm and so cannot be avoided
        easily. It requires extremely large-scale correlations to be included
        in the correlation matrices. These correlation matrices are inverted
        by the algorithm, leading to ill-defined inversions.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the incoming wavefront is defined.
    Cn_squared : scalar
        The integrated strength of the turbulence for this layer.
    L0 : scalar
        The outer scale for the atmospheric turbulence of this layer.
        The default is infinity.
    velocity : scalar or array_like
        The wind speed for this atmospheric layer. If this is a scalar,
        the wind will be along x. If this is a 2D array, then the values
        are interpreted as the wind speed along x and y. The default is
        zero.
    height : scalar
        The height of the atmospheric layer. By itself, this value has no
        influence, but it'll be used by the AtmosphericModel to perform
        inter-layer propagations.
    stencil_length : integer
        The number of columns/rows to use for the extrapolation of the phase
        along a column/row. Additionally, the stencil will contain one more
        pixel a little away from these initial columns/rows. Note, do not
        modify this parameter unless you know what you are doing. This parameter
        can be unintuitive. The default value of 2 is usually sufficient in
        most instances.
    use_interpolation : boolean
        whether to use sub-pixel interpolation of the phase screen. Bilinear
        interpolation is used. Without interpolation, for short integration times
        or fast loop speeds, the phase screen may stay on the same pixel for
        multiple frames. With interpolation, these discrete pixel transitions
        are smoothed out. The default is True.
    seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
        A seed to initialize the spectral noise. If None, then fresh, unpredictable
        entry will be pulled from the OS. If an int or array of ints, then it will
        be passed to a numpy.SeedSequency to derive the initial BitGenerator state.
        If a BitGenerator or Generator are passed, these will be wrapped and used
        instead. Default: None.

    Raises
    ------
    ValueError
        When the input grid is not cartesian, regularly spaced and two-dimensional.
    '''
    def __init__(self, input_grid, Cn_squared, L0=np.inf, velocity=0, height=0, stencil_length=2, use_interpolation=True, seed=None):
        self._initialized = False

        AtmosphericLayer.__init__(self, input_grid, Cn_squared, L0, velocity, height)

        # Check properties of input_grid
        if not input_grid.is_('cartesian'):
            raise ValueError('Input grid must be cartesian.')
        if not input_grid.is_regular:
            raise ValueError('Input grid must be regularly spaced')
        if not input_grid.ndim == 2:
            raise ValueError('Input grid must be two-dimensional.')

        self.stencil_length = stencil_length
        self.use_interpolation = use_interpolation
        self.rng = np.random.default_rng(seed)

        self._make_stencils()
        self._make_covariance_matrices()
        self._make_ab_matrices()

        self._original_rng = self.rng
        self.reset()

        self._initialized = True

    def _recalculate_matrices(self):
        if self._initialized:
            self._make_covariance_matrices()
            self._make_ab_matrices()

    def _make_stencils(self):
        # Vertical
        zero = self.input_grid.zero - np.array([0, self.input_grid.delta[1]])
        self.new_grid_bottom = CartesianGrid(RegularCoords(self.input_grid.delta, [self.input_grid.dims[0], 1], zero))

        self.stencil_bottom = Field(np.zeros(self.input_grid.size, dtype='bool'), self.input_grid).shaped
        self.stencil_bottom[:self.stencil_length, :] = True

        for i, n in enumerate(self.rng.geometric(0.5, self.input_grid.dims[0])):
            self.stencil_bottom[(n + self.stencil_length - 1) % self.input_grid.dims[1], i] = True

        self.stencil_bottom = self.stencil_bottom.ravel()
        self.num_stencils_vertical = np.sum(self.stencil_bottom)

        # Horizontal
        zero = self.input_grid.zero - np.array([self.input_grid.delta[0], 0])
        self.new_grid_left = CartesianGrid(RegularCoords(self.input_grid.delta, [1, self.input_grid.dims[1]], zero))

        self.stencil_left = Field(np.zeros(self.input_grid.size, dtype='bool'), self.input_grid).shaped
        self.stencil_left[:, :self.stencil_length] = True

        for i, n in enumerate(self.rng.geometric(0.5, self.input_grid.dims[1])):
            self.stencil_left[i, (n + self.stencil_length - 1) % self.input_grid.dims[0]] = True

        self.stencil_left = self.stencil_left.ravel()
        self.num_stencils_horizontal = np.sum(self.stencil_left)

    def _make_covariance_matrices(self):
        phase_covariance = phase_covariance_von_karman(fried_parameter_from_Cn_squared(1, 1), self.L0)

        # Vertical
        x = np.concatenate((self.input_grid.x[self.stencil_bottom], self.new_grid_bottom.x))
        x = np.concatenate([x - xx for xx in x])
        y = np.concatenate((self.input_grid.y[self.stencil_bottom], self.new_grid_bottom.y))
        y = np.concatenate([y - yy for yy in y])

        separations = CartesianGrid(UnstructuredCoords((x, y)))
        n = self.new_grid_bottom.size + self.num_stencils_vertical
        self.cov_matrix_vertical = phase_covariance(separations).reshape((n, n))

        # Horizontal
        x = np.concatenate((self.input_grid.x[self.stencil_left], self.new_grid_left.x))
        x = np.concatenate([x - xx for xx in x])
        y = np.concatenate((self.input_grid.y[self.stencil_left], self.new_grid_left.y))
        y = np.concatenate([y - yy for yy in y])

        separations = CartesianGrid(UnstructuredCoords((x, y)))
        n = self.new_grid_left.size + self.num_stencils_horizontal
        self.cov_matrix_horizontal = phase_covariance(separations).reshape((n, n))

    def _make_ab_matrices(self):
        # Vertical
        n = self.num_stencils_vertical
        cov_zz = self.cov_matrix_vertical[:n, :n]
        cov_xz = self.cov_matrix_vertical[n:, :n]
        cov_zx = self.cov_matrix_vertical[:n, n:]
        cov_xx = self.cov_matrix_vertical[n:, n:]

        cf = linalg.cho_factor(cov_zz)
        inv_cov_zz = linalg.cho_solve(cf, np.eye(cov_zz.shape[0]))

        self.A_vertical = cov_xz.dot(inv_cov_zz)

        BBt = cov_xx - self.A_vertical.dot(cov_zx)

        U, S, Vt = np.linalg.svd(BBt)
        L = np.sqrt(S[:self.input_grid.dims[0]])

        self.B_vertical = U * L

        # Horizontal
        n = self.num_stencils_horizontal
        cov_zz = self.cov_matrix_horizontal[:n, :n]
        cov_xz = self.cov_matrix_horizontal[n:, :n]
        cov_zx = self.cov_matrix_horizontal[:n, n:]
        cov_xx = self.cov_matrix_horizontal[n:, n:]

        cf = linalg.cho_factor(cov_zz)
        inv_cov_zz = linalg.cho_solve(cf, np.eye(cov_zz.shape[0]))

        self.A_horizontal = cov_xz.dot(inv_cov_zz)

        BBt = cov_xx - self.A_horizontal.dot(cov_zx)

        U, S, Vt = np.linalg.svd(BBt)
        L = np.sqrt(S[:self.input_grid.dims[1]])

        self.B_horizontal = U * L

    def _make_initial_phase_screen(self):
        oversampling = 16

        layer = FiniteAtmosphericLayer(self.input_grid, self.Cn_squared, self.outer_scale, self.velocity, self.height, oversampling, self.rng)

        self._achromatic_screen = layer.phase_for(1)
        self._shifted_achromatic_screen = self._achromatic_screen

        # Reuse the rng from the temporary layer, since it "forwarded" the randomness.
        # This avoids reusing the same randomness every call to reset().
        self.rng = layer.rng

    def _extrude(self, where=None):
        flipped = (where == 'top') or (where == 'right')
        horizontal = (where == 'left') or (where == 'right')

        if where == 'top' or where == 'right':
            screen = self._achromatic_screen[::-1]
        else:
            screen = self._achromatic_screen

        if horizontal:
            stencil = self.stencil_left
            A = self.A_horizontal
            B = self.B_horizontal
        else:
            stencil = self.stencil_bottom
            A = self.A_vertical
            B = self.B_vertical

        stencil_data = screen[stencil]
        random_data = self.rng.normal(0, 1, size=B.shape[1])
        new_slice = A.dot(stencil_data) + B.dot(random_data) * np.sqrt(self._Cn_squared)

        screen = screen.shaped

        if horizontal:
            screen = np.hstack((new_slice[:, np.newaxis], screen[:, :-1]))
        else:
            screen = np.vstack((new_slice[np.newaxis, :], screen[:-1, :]))

        screen = Field(screen, self.input_grid)

        if flipped:
            self._achromatic_screen = screen[::-1, ::-1].ravel()
        else:
            self._achromatic_screen = screen.ravel()

    def phase_for(self, wavelength):
        '''Compute the phase at a certain wavelength.

        Parameters
        ----------
        wavelength : scalar
            The wavelength of the light for which to compute the phase screen.

        Returns
        -------
        Field
            The computed phase screen.
        '''
        return self._shifted_achromatic_screen / wavelength

    def reset(self, make_independent_realization=False):
        '''Reset the atmospheric layer to t=0.

        Parameters
        ----------
        make_independent_realization : boolean
            Whether to start an independent realization of the noise for the
            atmospheric layer or not. When this is False, the exact same phase
            screens will be generated as in the first run, as long as the Cn^2
            and outer scale are the same. This allows for testing of multiple
            runs of a control system with different control parameters. When
            this is True, an independent realization of the atmospheric layer
            will be generated. This is useful for Monte-Carlo-style computations.
            The default is False.
        '''
        if make_independent_realization:
            # Reset the original random generator to the current one. This
            # will essentially reset the randomness.
            self._original_rng = copy.copy(self.rng)
        else:
            # Make a copy of the original random generator. This copy will be
            # used as the source for all randomness.
            self.rng = copy.copy(self._original_rng)

        self._make_initial_phase_screen()

        self.center = np.zeros(2)
        self._t = 0

    def evolve_until(self, t):
        '''Evolve the atmospheric layer until a certain time.

        .. note::
            Backwards evolution is not allowed. The information about previous
            parts of the phase screen are inherently lost. If you want to replay the
            same atmospheric noise again, please use `reset()` to reset the layer
            to its starting position.

        Parameters
        ----------
        t : scalar
            The new time to evolve the phase screen to. This should be larger or equal to
            the current time of the atmospheric layer.

        Raises
        ------
        ValueError
            When the time `t` is smaller than the current time of the layer.
        '''
        if t is None:
            self.reset()
            return

        if t < self._t:
            raise ValueError('Backwards temporal evolution is not allowed.')

        # Get the old and new center positions
        old_center = np.round(self.center / self.input_grid.delta).astype('int')
        self.center = self.center + self.velocity * (t - self._t)
        new_pixel_center = np.round(self.center / self.input_grid.delta).astype('int')

        self._t = t

        # Measure the number of full pixel shifts
        delta = new_pixel_center - old_center
        # Measure the sub-pixel shift
        sub_delta = self.center - new_pixel_center * self.input_grid.delta

        for i in range(abs(delta[0])):
            if delta[0] < 0:
                self._extrude('left')
            else:
                self._extrude('right')

        for i in range(abs(delta[1])):
            if delta[1] < 0:
                self._extrude('bottom')
            else:
                self._extrude('top')

        if self.use_interpolation:
            # Use bilinear interpolation to interpolate the achromatic phase screen to the correct position.
            # This is to avoid sudden shifts by discrete pixels.
            ps = self._achromatic_screen.shaped

            with warnings.catch_warnings():
                # Suppress warnings about the changed behaviour in affine_transform for 1D arrays.
                # We know about this and are expecting the behaviour as is.
                warnings.filterwarnings('ignore', message='The behaviour of affine_transform')
                warnings.filterwarnings('ignore', message='The behavior of affine_transform')

                screen = affine_transform(ps, np.array([1, 1]), (sub_delta / self.input_grid.delta)[::-1], mode='nearest', order=5)
                self._shifted_achromatic_screen = Field(screen.ravel(), self._achromatic_screen.grid)
        else:
            self._shifted_achromatic_screen = self._achromatic_screen

    @property
    def Cn_squared(self):  # noqa: N802
        '''The integrated strength of the turbulence for this layer.
        '''
        return self._Cn_squared

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):  # noqa: N802
        self._Cn_squared = Cn_squared

    @property
    def outer_scale(self):
        '''The outer scale of the turbulence for this layer.
        '''
        return self._L0

    @outer_scale.setter
    def L0(self, L0):  # noqa: N802
        self._L0 = L0

        self._recalculate_matrices()
