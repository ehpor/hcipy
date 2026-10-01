import numpy as np
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
            order = int(2 * np.ceil(0.5 * (np.sqrt(8 * len(coeffs) + 1) - 1)))
            self.coeffs = coeffs
        else:
            self.coeffs = np.ones(int(order * (order / 2 + 1) / 4))

        self._p = order // 2

        # For order 2 there is only a single mode, for which the separable path is slower than
        # simply applying the full mode projection.
        self._use_separated_path = (
            order > 2
            and self.pupil_grid.is_separated
            and self.pupil_grid.ndim == 2
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
        x, y = self.pupil_grid.separated_coords
        apv = self._aperture.shaped

        # Triangular mode indices (j, k) with x^j y^k of total degree j + k < p.
        triangular = [(j, i - j) for i in range(self._p) for j in range(i + 1)]
        tri_j = np.array([t[0] for t in triangular])
        tri_k = np.array([t[1] for t in triangular])

        X = x[:, np.newaxis] ** np.arange(self._p)
        Y = y[:, np.newaxis] ** np.arange(self._p)

        dmax = 2 * self._p - 1
        Xp = x[:, np.newaxis] ** np.arange(dmax)
        Yp = y[:, np.newaxis] ** np.arange(dmax)

        T = Xp.T.dot((np.abs(apv) ** 2).T).dot(Yp)

        # G[a, b] = <mode_a, mode_b> = T[j_a + j_b, k_a + k_b].
        j = tri_j[:, np.newaxis] + tri_j[np.newaxis, :]
        k = tri_k[:, np.newaxis] + tri_k[np.newaxis, :]
        G = T[j, k]

        try:
            R = np.linalg.cholesky(G).T
            R_inverse = np.linalg.inv(R)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                'The Gram matrix of the pupil modes is not positive definite, which means the modes are '
                'linearly dependent on this aperture. This typically happens when the coronagraph order is '
                'too high for the aperture support or the pupil is undersampled. Reduce the order or '
                'increase the pupil sampling.'
            ) from e

        K = R_inverse.dot(np.diag(self.coeffs)).dot(R_inverse.conj().T)

        # Embed the triangular correction into a full p^2 x p^2 matrix acting on the flattened
        # p x p coefficient matrix. Its rows and columns outside the triangle are zero, so the
        # reconstructed coefficient matrix is automatically zero there as well.
        flat = tri_j * self._p + tri_k
        self._K_full = np.zeros((self._p**2, self._p**2), dtype=K.dtype)
        self._K_full[flat[:, np.newaxis], flat[np.newaxis, :]] = K
        self._flat = flat

        self._X = X
        self._Y = Y

    def _make_modes(self):
        '''Construct the explicit pupil modes :math:`x^j y^k` of total degree less than ``p``.'''
        return [
            self._aperture * self.pupil_grid.x ** j * self.pupil_grid.y ** (i - j)
            for i in range(self._p)
            for j in range(i + 1)
        ]

    def _setup_nonseparated(self):
        '''Set up the explicit mode matrix for non-separated grids (and order 2).'''
        mode_basis = ModeBasis(self._make_modes()).orthogonalized

        self._transformation = mode_basis.transformation_matrix
        self._transformation_inverse = inverse_truncated(self._transformation, 1e-6)

    def _explicit_mode_matrix(self):
        '''Construct the explicit (non-orthogonalized) mode matrix.

        This is only used for the rarely-called transformation matrix methods, so it is built
        on demand rather than stored.
        '''
        return np.stack([np.asarray(mode) for mode in self._make_modes()], axis=-1)

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
            electric_field = wf.electric_field
            leading_shape = electric_field.shape[:-1]
            aperture = self._aperture.shaped

            # Project the wavefront onto the pupil modes using the separability of the modes.
            weighted = aperture.conj() * electric_field.reshape(leading_shape + aperture.shape)
            coefficients = np.einsum('...li,ij,lk->...jk', weighted, self._X, self._Y, optimize=True)

            # Correct the coefficients with the pre-calculated inverse Gram matrix to get the
            # actual coefficients.
            coefficients = np.einsum('mn,...n->...m', self._K_full, coefficients.reshape(leading_shape + (self._p**2,)), optimize=True)
            coefficient_matrix = coefficients.reshape(leading_shape + (self._p, self._p))

            # Transform back to the pupil.
            correction = aperture * np.einsum('...jk,ij,lk->...li', coefficient_matrix, self._X, self._Y, optimize=True)

            wf.electric_field -= correction.reshape(electric_field.shape)
        else:
            # Project the wavefront onto the pupil modes and attenuate by coeffs, then transform back to the pupil.
            # This is done for each polarization separately.
            correction = np.einsum(
                'kj,j,ji,...i->...k',
                self._transformation,
                self.coeffs,
                self._transformation_inverse,
                wf.electric_field,
                optimize='optimal'
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
        if self._use_separated_path:
            mode_matrix = self._explicit_mode_matrix()
            k = self._K_full[self._flat[:, np.newaxis], self._flat[np.newaxis, :]]
            return np.eye(self.pupil_grid.size) - mode_matrix.dot(k.dot(mode_matrix.conj().T))

        return np.eye(self.pupil_grid.size) - self._transformation.dot(np.expand_dims(self.coeffs, axis=1) * self._transformation_inverse)

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
