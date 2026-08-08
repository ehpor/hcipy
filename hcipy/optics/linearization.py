from dataclasses import dataclass

import numpy as np

from ..field import Field, FieldBase
from ..mode_basis import ModeBasis
from .wavefront import Wavefront


@dataclass
class LinearizedOpticalSystem:
    '''The first-order (linearized) response of an optical system to wavefront error modes.

    This model describes the output electric field of an optical system to
    first order in the coefficients `phi` of a set of wavefront error modes:

    `E_out = static_field + field_response * phi`

    Parameters
    ----------
    static_field : Field
        The complex electric field on the output grid for zero mode coefficients (E_0).
    field_response : ModeBasis
        The per-mode response fields on the output grid (G). The columns of this
        basis are the derivatives of the output electric field with respect to the
        mode coefficients.
    modes : ModeBasis
        The input wavefront error modes, as passed to
        :func:`linearize_optical_system`.
    wavelength : scalar
        The wavelength for which the model was linearized.
    mode_type : str
        The semantics of the modes ('opd', 'phase', 'amplitude' or 'field'),
        documenting the units of the modes and of the mode coefficients.
    '''
    static_field: FieldBase
    field_response: ModeBasis
    modes: ModeBasis
    wavelength: float
    mode_type: str

    @property
    def input_grid(self):
        '''The grid on which the modes are defined.
        '''
        return self.modes.grid

    @property
    def output_grid(self):
        '''The grid on which the output electric field is defined.
        '''
        return self.static_field.grid

    def __call__(self, coefficients):
        '''Evaluate the linearized output electric field for given mode coefficients.

        This evaluates :math:`E = E_0 + G * \\phi` on the output grid, using a single
        matrix-vector product without intermediate copies of the response basis.

        Parameters
        ----------
        coefficients : ndarray
            The mode coefficients `phi`, with one entry per mode.

        Returns
        -------
        Field
            The output electric field on the output grid.

        Raises
        ------
        ValueError
            If the number of coefficients does not match the number of modes.
        '''
        coefficients = np.asarray(coefficients)

        if coefficients.shape != (self.modes.num_modes,):
            raise ValueError('The number of coefficients must match the number of modes.')

        return self.static_field + self.field_response.transformation_matrix @ coefficients

    def intensity_from_covariance(self, P):
        '''The mean intensity on the output grid for mode coefficients with covariance `P`.

        For mode coefficients `phi` with zero mean and covariance `P`, the mean
        intensity of the linearized field is:

        .. math::

            E[I] = |E_0|^2 + diag(G P G^H)

        If `P` is given as a vector, it is interpreted as the diagonal of the
        covariance matrix (the per-mode variances). The diagonal of :math:`G P G^H` is
        computed without materializing the `(n_out, n_out)` covariance matrix of
        the output field, requiring only `(n_out, r)` storage.

        Parameters
        ----------
        P : ndarray
            The covariance matrix of the mode coefficients, of shape `(r, r)`,
            or a vector of length `r` containing its diagonal.

        Returns
        -------
        Field
            The mean intensity on the output grid.

        Raises
        ------
        ValueError
            If `P` is not a vector of length `r` or a matrix of shape `(r, r)`.
        '''
        P = np.asarray(P)

        if P.ndim == 1:
            if P.shape != (self.modes.num_modes,):
                raise ValueError('A vector P must have one entry per mode.')

            # diag(G diag(P) G^H) = |G|^2 P, a single matrix-vector product.
            variance = (np.abs(self.field_response.transformation_matrix)**2) @ P
        elif P.ndim == 2:
            if P.shape != (self.modes.num_modes, self.modes.num_modes):
                raise ValueError('A matrix P must have shape (num_modes, num_modes).')

            # diag(G P G^H) = sum_j (G P)_ij conj(G)_ij, computed via one
            # (n_out, r) intermediate instead of the full (n_out, n_out) matrix.
            G = self.field_response.transformation_matrix
            variance = np.sum((G @ P) * np.conj(G), axis=1).real
        else:
            raise ValueError('P must be a vector of length r or a matrix of shape (r, r).')

        return Field(np.abs(self.static_field)**2 + variance, self.output_grid)

    def pastis_matrix(self, weights):
        '''The PASTIS matrix of this linearized optical system, weighted per output point.

        The PASTIS matrix describes the second-order response of the weighted
        sum of the intensity on the output grid to the mode coefficients:

        .. math::

            M = G^H W G,

        with ::math::`W = diag(weights)`. See [Leboulleux2018]_.

        .. [Leboulleux2018]

            L. Leboulleux, J.-F. Sauvage, L. A. Pueyo, T. Fusco, R. Soummer,
            J. Mazoyer, A. Sivaramakrishnan, M. N'Diaye, and O. Fauvarque, "Pair-based Analytical
            model for Segmented Telescopes Imaging from Space (PASTIS) for sensitivity analysis,"
            Journal of Astronomical Telescopes, Instruments, and Systems, 4(3), 035002 (2018).
            https://doi.org/10.1117/1.JATIS.4.3.035002

        The contribution of a mode-coefficient covariance `P` to the weighted
        intensity is `trace(M P)`. The matrix is Hermitian; for real symmetric
        `P`, `trace(M P)` is real and the real part of `M` follows the usual
        PASTIS convention. It is computed with `(n_out, r)` storage, without
        materializing the weight matrix.

        Parameters
        ----------
        weights : Field or ndarray
            The weights for each point on the output grid, e.g. a dark hole mask.

        Returns
        -------
        ndarray
            The weighted PASTIS matrix, of shape `(r, r)`.

        Raises
        ------
        ValueError
            If the weights do not have one entry per point on the output grid.
        '''
        weights = np.asarray(weights)

        if weights.shape != (self.output_grid.size,):
            raise ValueError('The weights must have one entry per point on the output grid.')

        # M = G^H W G = (W G)^H G, with the weight matrix applied as a
        # column-wise scaling of the response basis.
        G = self.field_response.transformation_matrix

        return np.asarray((G * weights[:, np.newaxis]).conj().T @ G)


def linearize_optical_system(optical_system, aperture, modes, wavelength=1, mode_type='opd'):
    '''Linearize an optical system with respect to wavefront error modes.

    This function computes the first-order response of an optical system to a
    set of wavefront error (WFE) modes: `E_out = E_0 + G * phi`, where `phi`
    are the WFE-mode coefficients and `G` is the complex electric-field
    sensitivity of the output plane to each mode.

    The linearization is exact: for each mode, a single propagation is used to
    compute the corresponding response field, with no step size and no finite
    differences, exploiting the complex-linearity of the optical system.

    Parameters
    ----------
    optical_system : OpticalElement or callable
        The optical system to linearize. This can be anything callable that
        maps a :class:`Wavefront` to a :class:`Wavefront`. Since
        `OpticalElement.__call__` is aliased to `forward`, an
        :class:`OpticalSystem` can be passed as-is. The system must be
        complex-linear in the electric field; all standard HCIPy optical
        elements are.
    aperture : Field or callable
        The entrance electric field `A`, as a :class:`Field` or a callable
        (e.g. an HCIPy aperture generator) evaluated on the input grid. The
        field may be complex.
    modes : ModeBasis
        The wavefront error modes. Their semantics are defined by `mode_type`.
    wavelength : scalar
        The wavelength for which the model is linearized.
    mode_type : str
        The semantics of the modes, and hence the units of the mode
        coefficients and of the returned response:

        ==============  ==============================================  =====================================
        `mode_type`     Semantics of a mode `m_j`                      Response column
        ==============  ==============================================  =====================================
        'opd'           Optical path difference shape, in meters       `G[:, j] = 1j * k * F(A * m_j)`
        'phase'         Phase shape, in radians                        `G[:, j] = 1j * F(A * m_j)`
        'amplitude'     Dimensionless multiplicative amplitude         `G[:, j] = F(A * m_j)`
        'field'         Additive complex field perturbation, with      `G[:, j] = F(m_j)`
                        the support included in `m_j`
        ==============  ==============================================  =====================================

        with `k = 2 * pi / wavelength`.

    Returns
    -------
    LinearizedOpticalSystem
        The linearized optical system, carrying the static field `E_0`, the
        response basis `G`, the modes, the wavelength and the mode type.

    Raises
    ------
    ValueError
        If `mode_type` is not one of the supported values, or if no input grid
        can be derived from the modes or the aperture, or if the grids of the
        modes and the aperture do not match.
    TypeError
        If `optical_system` is not callable.
    '''
    if not callable(optical_system):
        raise TypeError('The optical system must be callable with a Wavefront.')

    if mode_type not in ('opd', 'phase', 'amplitude', 'field'):
        raise ValueError('Unknown mode type %r. Mode type must be one of "opd", "phase", "amplitude" or "field".' % mode_type)

    if isinstance(aperture, Field):
        aperture_grid = aperture.grid
    else:
        aperture_grid = None

    modes_grid = getattr(modes, 'grid', None)

    if modes_grid is not None and aperture_grid is not None and modes_grid != aperture_grid:
        raise ValueError('The grid of the modes and the grid of the aperture must match.')

    input_grid = modes_grid if modes_grid is not None else aperture_grid

    if input_grid is None:
        raise ValueError('No input grid could be derived from the modes or the aperture. Supply modes with a grid, or an aperture Field.')

    if modes_grid is None and isinstance(modes, ModeBasis):
        # Attach the derived input grid to a copy of the modes, so that the
        # model carries the input grid even when the modes were given without one.
        modes = ModeBasis(modes.transformation_matrix.copy(), input_grid)

    if isinstance(aperture, Field):
        aperture_field = aperture
    else:
        aperture_field = aperture(input_grid)
        if not isinstance(aperture_field, Field):
            raise ValueError('The aperture callable did not return a Field.')

    if not aperture_field.is_valid_field:
        raise ValueError('The aperture field does not have the correct size for its grid.')

    # The unperturbed propagation. This also discovers the output grid.
    static_field = optical_system(Wavefront(aperture_field, wavelength)).electric_field

    if mode_type == 'opd':
        factor = 1j * 2 * np.pi / wavelength
    elif mode_type == 'phase':
        factor = 1j
    else:
        factor = 1

    response_columns = []
    for j in range(len(modes)):
        mode = modes[j]

        if not isinstance(mode, Field):
            mode = Field(np.asarray(mode), input_grid)

        if mode_type == 'field':
            input_field = mode
        else:
            input_field = aperture_field * mode

        response = optical_system(Wavefront(input_field, wavelength)).electric_field
        response_columns.append(factor * response)

    field_response = ModeBasis(np.stack(response_columns, axis=-1), static_field.grid)

    return LinearizedOpticalSystem(static_field, field_response, modes, wavelength, mode_type)
