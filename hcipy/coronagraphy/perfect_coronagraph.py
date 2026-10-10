import math
from .._math.einsum import einsum
from ..field import NewStyleField
from ..optics import OpticalElement
from ..mode_basis import ModeBasis
from ..util import inverse_truncated

class PerfectCoronagraph(OpticalElement):
    r'''A perfect coronagraph for a certain aperture and order.

    These type of coronagraphs suppress all light for a flat wavefront. The incoming complex
    amplitude :math:`A` is modified as follows (following [Cavarroc2006]_):

    .. math::
        \overline{A} = A - \Pi \sqrt{S}

    where :math:`\overline{A}` is the resulting complex ampliutude, :math:`\Pi` is the telescope
    pupil, and :math:`S` is the Strehl ratio of the incoming wavefront.

    Higher orders are added by fitting higher-order electric field modes to the incoming
    wavefront and subtracting those, following [Guyon2006]_.

    .. [Cavarroc2006] Celine Cavarroc et al. "Fundamental limitations on Earth-like planet detection with
        extremely large telescopes." Astronomy & Astrophysics 447.1 (2006): 397-403

    .. [Guyon2006] Olivier Guyon et al. "Theoretical limits on extrasolar terrestrial planet detection
        with coronagraphs." The Astrophysical Journal Supplement Series 167.1 (2006): 81

    Parameters
    ----------
    aperture : Field
        The reference aperture. The perfect coronagraph is designed for this aperture.
    order : integer
        The order of the perfect coronagraph. This must be even.
    coeffs : list or ndarray or None
        The coefficients that are used for subtraction. This allows for partial suppression of certain
        modes, which can be used to design perfect coronagraphs that are insensitive to stellar
        radius [Guyon2006]_. If it is None, all modes are completely suppressed.
    '''
    def __init__(self, aperture, order=2, coeffs=None):
        assert order % 2 == 0, "The coronagraph order has to be even."

        self.pupil_grid = aperture.grid
        self._aperture = aperture

        if coeffs is not None:
            order = int(2 * math.ceil(0.5 * (math.sqrt(8 * len(coeffs) + 1) - 1)))
            self.coeffs = coeffs
        else:
            self.coeffs = self.pupil_grid.xp.ones(int(order * (order / 2 + 1) / 4))

        self._p = order // 2

        # Can we use the separated grid path? Grid needs to be 2D and
        # separated. Also for order 2, there is only one mode, so
        # the separated path is slower.
        self._use_separated_path = (
            self.pupil_grid.is_separated
            and self.pupil_grid.ndim == 2
            and order > 2
        )

        if self._use_separated_path:
            self._setup_separated()
        else:
            self._setup_nonseparated()

    def _setup_separated(self):
        '''Precompute the separable projection operators.

        This sets up the two-dimensional power matrices for the x- and y-coordinates, as well as
        the inverse-Gram correction matrix.
        '''
        xp = self.pupil_grid.xp
        x, y = self.pupil_grid.separated_coords

        # Build {1, x, x^2, ...} and same for y.
        X = x[:, xp.newaxis]**xp.arange(self._p, dtype=x.dtype)
        Y = y[:, xp.newaxis]**xp.arange(self._p, dtype=y.dtype)

        # We need higher powers of x and y to construct the Gram matrix.
        dmax = 2 * self._p - 1
        Xp = x[:, xp.newaxis]**xp.arange(dmax, dtype=x.dtype)
        Yp = y[:, xp.newaxis]**xp.arange(dmax, dtype=y.dtype)

        # Computed all weighted dot products.
        aperture_shaped = self._aperture.shaped
        aperture_shaped = aperture_shaped.data if isinstance(aperture_shaped, NewStyleField) else aperture_shaped

        T = xp.matmul(xp.matrix_transpose(Xp), xp.matmul(xp.matrix_transpose(xp.abs(aperture_shaped)**2), Yp))

        # Triangular mode indices (j, k) with x^j y^k of total degree j + k < p.
        j = xp.asarray([j for i in range(self._p) for j in range(i + 1)])
        k = xp.asarray([i - j for i in range(self._p) for j in range(i + 1)])

        # G[a, b] = <mode_a, mode_b> = T[j_a + j_b, k_a + k_b].
        G = T[j[:, xp.newaxis] + j[xp.newaxis, :], k[:, xp.newaxis] + k[xp.newaxis, :]]

        # Compute the lower Cholesky factor L with G = L L^H.
        try:
            L = xp.linalg.cholesky(G)

            # Some backends (Jax) return NaNs instead of raising for a non-positive-definite matrix.
            if not bool(xp.all(xp.isfinite(L))):
                raise ValueError('Cholesky returned indeterminate result.')
        except Exception as e:
            raise ValueError(
                'The Gram matrix of the pupil modes is not positive definite, which means the modes are '
                'linearly dependent on this aperture. This typically happens when the coronagraph order is '
                'too high for the aperture support or the pupil is undersampled. Reduce the order or '
                'increase the pupil sampling.'
            ) from e

        # K = L^-H diag(coeffs) L^-1.
        L_inverse = xp.linalg.inv(L)
        coeffs = xp.asarray(self.coeffs)
        self._K = xp.matmul(xp.matrix_transpose(xp.conj(L_inverse)) * coeffs[xp.newaxis, :], L_inverse)

        # Embed the triangular correction into a full p^2 x p^2 matrix acting on the flattened
        # p x p coefficient matrix.
        flat = j * self._p + k
        selection = xp.take(xp.eye(self._p**2, dtype=self._K.dtype), flat, axis=0)
        self._K_full = xp.matmul(xp.matrix_transpose(selection), xp.matmul(self._K, selection))

        self._X = X
        self._Y = Y

    def _make_modes(self):
        '''Construct the explicit pupil modes :math:`x^j y^k` of total degree less than ``p``.'''
        return [
            self._aperture * self.pupil_grid.x**j * self.pupil_grid.y**(i - j)
            for i in range(self._p)
            for j in range(i + 1)
        ]

    def _setup_nonseparated(self):
        '''Set up the explicit mode matrix for non-separated grids (and order 2).'''
        mode_basis = ModeBasis(self._make_modes()).orthogonalized

        self._transformation = mode_basis.transformation_matrix
        self._transformation_inverse = inverse_truncated(self._transformation, 1e-6)

    def forward(self, wavefront):
        '''Propagate the wavefront through the perfect coronagraph.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront (in the pupil plane)

        Returns
        -------
        Wavefront
            The post-coronagraphic wavefront (in the pupil plane).
        '''
        wf = wavefront.copy()

        if self._use_separated_path:
            xp = self.pupil_grid.xp
            electric_field = wf.electric_field
            leading_shape = electric_field.shape[:-1]

            aperture = self._aperture.shaped
            if isinstance(aperture, NewStyleField):
                aperture = aperture.data

            # Project the wavefront onto the pupil modes using the separability of the modes.
            weighted = xp.conj(aperture) * xp.reshape(electric_field, leading_shape + aperture.shape)
            coefficients = einsum('...li,ij,lk->...jk', weighted, self._X, self._Y)

            # Attenuate the coefficients.
            coefficients = xp.reshape(coefficients, (-1, self._p**2))
            coefficients = xp.matmul(coefficients, self._K_full)
            coefficient_matrix = xp.reshape(coefficients, leading_shape + (self._p, self._p))

            # Transform back to the pupil.
            correction = aperture * einsum('...jk,ij,lk->...li', coefficient_matrix, self._X, self._Y)

            wf.electric_field -= xp.reshape(correction, electric_field.shape)
        else:
            # Project the wavefront onto the pupil modes and attenuate by coeffs, then transform back to the pupil.
            # This is done for each polarization separately.
            correction = einsum(
                'kj,j,ji,...i->...k',
                self._transformation,
                self.coeffs,
                self._transformation_inverse,
                wf.electric_field
            )

            # Then subtract this from the original wavefront to compute the post-coronagraphic field.
            wf.electric_field -= correction

        return wf

    def backward(self, wavefront):
        '''Propagate the wavefront backwards through the perfect coronagraph.

        This method behaves the same as the forward propagation.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront (in the pupil plane)

        Returns
        -------
        Wavefront
            The post-coronagraphic wavefront (in the pupil plane).
        '''
        return self.forward(wavefront)

    def get_transformation_matrix_forward(self, wavelength=1):
        '''Get the forward propagation transformation matrix.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the transformation matrix.

        Returns
        -------
        ndarray
            The forward transformation_matrix.
        '''
        xp = self.pupil_grid.xp

        if self._use_separated_path:
            # Build the explicit (non-orthogonalized) mode matrix on demand.
            modes = [mode.data if isinstance(mode, NewStyleField) else xp.asarray(mode) for mode in self._make_modes()]
            mode_matrix = xp.stack([xp.reshape(xp.asarray(mode), (-1,)) for mode in modes], axis=-1)

            k = self._K
            return xp.eye(self.pupil_grid.size, dtype=k.dtype) - xp.matmul(mode_matrix, xp.matmul(k, xp.matrix_transpose(xp.conj(mode_matrix))))

        return xp.eye(self.pupil_grid.size) - xp.matmul(self._transformation, self.coeffs[:, xp.newaxis] * self._transformation_inverse)

    def get_transformation_matrix_backward(self, wavelength=1):
        '''Get the backwards propagation transformation matrix.

        This method behaves the same as the forward propagation.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the transformation matrix.

        Returns
        -------
        ndarray
            The backward transformation_matrix.
        '''
        return self.get_transformation_matrix_forward(wavelength)
