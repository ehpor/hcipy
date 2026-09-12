from .._math.backends import infer_xp

__all__ = [
    'RayTransferMatrix',
    'make_abcd_free_space',
    'make_abcd_thin_lens',
    'make_abcd_mirror',
    'make_abcd_fraunhofer',
    'make_abcd_magnifier',
    'make_abcd_refractive_interface',
]

class RayTransferMatrix:
    '''A 2x2 ray-transfer (ABCD) matrix, optionally dispersive.

    Wraps either a fixed 2x2 array or a function of the wavelength (in
    meters) that returns one.

    Parameters
    ----------
    matrix : Array or callable
        The 2x2 matrix, or a callable ``matrix(wavelength)`` returning one.

    Examples
    --------
    >>> M = make_abcd_free_space(0.5)
    >>> M(500e-9)
    array([[1. , 0.5],
           [0. , 1. ]])
    '''
    __slots__ = ('_matrix')

    def __init__(self, matrix):
        if not callable(matrix):
            if matrix.shape != (2, 2):
                raise ValueError('RayTransferMatrix requires a 2x2 matrix.')

        self._matrix = matrix

    @property
    def is_dispersive(self):
        '''Whether the matrix depends on the wavelength.'''
        return callable(self._matrix)

    def __call__(self, wavelength):
        '''The 2x2 matrix evaluated at the given wavelength (in meters).'''
        if self.is_dispersive:
            return self._matrix(wavelength)
        return self._matrix

    def _elements(self, wavelength):
        '''The four matrix elements A, B, C, D as backend arrays.'''
        M = self(wavelength)
        return (M[0, 0], M[0, 1], M[1, 0], M[1, 1])

    def effective_focal_length(self, wavelength):
        '''The effective focal length f_eff = -1 / C of the ABCD system.'''
        _, _, c, _ = self._elements(wavelength)
        return -1.0 / c

    def paraxial_power(self, wavelength):
        '''The paraxial optical power Phi = -C of the ABCD system.'''
        _, _, c, _ = self._elements(wavelength)
        return -c

    def geometric_magnification(self, wavelength):
        '''The transverse geometric magnification m = A of the ABCD system.'''
        a, _, _, _ = self._elements(wavelength)
        return a

    def angular_magnification(self, wavelength):
        '''The angular magnification m_theta = D of the ABCD system.'''
        _, _, _, d = self._elements(wavelength)
        return d

    def distance(self, wavelength):
        '''The effective propagation distance z_eff = B of the ABCD system.'''
        _, b, _, _ = self._elements(wavelength)
        return b

    def front_focal_length(self, wavelength):
        '''The front focal length FFL = -A / C of the ABCD system.'''
        a, _, c, _ = self._elements(wavelength)
        return -a / c

    def back_focal_length(self, wavelength):
        '''The back focal length BFL = -D / C of the ABCD system.'''
        _, _, c, d = self._elements(wavelength)
        return -d / c

    def is_surface(self, wavelength, tol=1e-8):
        '''Whether the ABCD matrix is a pure powered surface (B ~ 0).'''
        _, b, _, _ = self._elements(wavelength)
        return abs(b) < tol

    def is_free_space(self, wavelength, tol=1e-8):
        '''Whether the ABCD matrix corresponds to free-space propagation.'''
        a, b, c, d = self._elements(wavelength)

        if abs(b) < tol:
            return False

        return (abs(a - 1.0) < tol) and (abs(c) < tol) and (abs(d - 1.0) < tol)

    def forward(self, beam):
        '''Transform a Gaussian beam through this matrix, returning a new beam.

        The complex beam parameter of the beam is transformed according to the
        ABCD law

            q' = (A q + B) / (C q + D).

        A new GaussianBeam is returned; the input beam is not modified. A
        dispersive matrix is evaluated at the wavelength of the beam.

        Parameters
        ----------
        beam : GaussianBeam
            The Gaussian beam to transform.

        Returns
        -------
        GaussianBeam
            The transformed Gaussian beam.
        '''
        return beam.propagate(self(beam.wavelength))

    def backward(self, beam):
        '''Inverse-transform a Gaussian beam through this matrix.

        The numerical inverse of the matrix is applied, so this works for any
        invertible matrix.

        Parameters
        ----------
        beam : GaussianBeam
            The Gaussian beam to transform.

        Returns
        -------
        GaussianBeam
            The inverse-transformed Gaussian beam.
        '''
        M = self(beam.wavelength)
        xp = infer_xp(M)

        return beam.propagate(xp.linalg.inv(M))

    def __matmul__(self, other):
        '''Compose two ray-transfer matrices: self is applied after other.'''
        other = other if isinstance(other, RayTransferMatrix) else RayTransferMatrix(other)

        if not (self.is_dispersive or other.is_dispersive):
            return RayTransferMatrix(self._matrix @ other._matrix)

        def composition(wavelength):
            return self(wavelength) @ other(wavelength)

        return RayTransferMatrix(composition)

    def __repr__(self):
        if self.is_dispersive:
            return 'RayTransferMatrix(dispersive)'
        return f'RayTransferMatrix({self._matrix!r})'

def make_abcd_free_space(distance, xp=None):
    '''The ABCD (ray-transfer) matrix for free-space propagation.

    Parameters
    ----------
    distance : scalar
        The propagation distance.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for free-space propagation.
    '''
    if xp is None:
        xp = infer_xp(distance)

    return RayTransferMatrix(xp.asarray([[1.0, distance], [0.0, 1.0]]))

def make_abcd_thin_lens(focal_length, xp=None):
    '''The ABCD (ray-transfer) matrix for a thin refractive lens in air.

    Parameters
    ----------
    focal_length : scalar
        The focal length of the lens.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for a thin lens.
    '''
    if xp is None:
        xp = infer_xp(focal_length)

    return RayTransferMatrix(xp.asarray([[1.0, 0.0], [-1.0 / focal_length, 1.0]]))

def make_abcd_mirror(radius=None, xp=None):
    '''The ABCD (ray-transfer) matrix for a spherical mirror.

    A radius of curvature of None (the default) corresponds to a plane
    mirror.

    Parameters
    ----------
    radius : scalar, optional
        The radius of curvature of the mirror. A positive radius corresponds
        to a concave (focusing) mirror. If None (default), the mirror is
        plane.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for a spherical mirror.
    '''
    if xp is None:
        xp = infer_xp(radius)

    if radius is None:
        return RayTransferMatrix(xp.asarray([[1.0, 0.0], [0.0, 1.0]]))

    return RayTransferMatrix(xp.asarray([[1.0, 0.0], [-2.0 / radius, 1.0]]))

def make_abcd_fraunhofer(focal_length, xp=None):
    '''The ABCD (ray-transfer) matrix for an ideal Fraunhofer (Fourier) transform.

    This matrix describes the propagation between two conjugate planes, giving
    the classic 'pupil-focal plane' Fourier relationship.

    Parameters
    ----------
    focal_length : scalar
        The effective focal length that defines the conjugate-plane transform.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for a Fraunhofer (Fourier) transform.
    '''
    if xp is None:
        xp = infer_xp(focal_length)

    return RayTransferMatrix(xp.asarray([[0.0, focal_length], [-1.0 / focal_length, 0.0]]))

def make_abcd_magnifier(magnification, xp=None):
    '''The ABCD (ray-transfer) matrix for an ideal magnifier (afocal telescope).

    Parameters
    ----------
    magnification : scalar
        The transverse geometric magnification of the system.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for an ideal magnifier.
    '''
    if xp is None:
        xp = infer_xp(magnification)

    return RayTransferMatrix(xp.asarray([[magnification, 0.0], [0.0, 1.0 / magnification]]))

def make_abcd_refractive_interface(refractive_index_in, refractive_index_out, radius=None, xp=None):
    '''The ABCD (ray-transfer) matrix for refraction at a dielectric interface.

    A positive radius corresponds to a surface whose center of curvature lies
    to the right of the interface. A radius of None (the default) corresponds
    to a flat interface. Note that this matrix has determinant
    refractive_index_in / refractive_index_out, since it accounts for the
    change of the ray angle due to refraction.

    A dispersive matrix is returned if either refractive index is a callable
    function of the wavelength (in meters).

    Parameters
    ----------
    refractive_index_in : scalar or callable
        The refractive index of the medium the ray is coming from, or a
        function of the wavelength (in meters) returning it.
    refractive_index_out : scalar or callable
        The refractive index of the medium the ray is going to, or a function
        of the wavelength (in meters) returning it.
    radius : scalar, optional
        The radius of curvature of the interface. If None (default), the
        interface is flat.
    xp : module, optional
        The Array API namespace to use. If not given, it will be inferred from
        the other parameters.

    Returns
    -------
    RayTransferMatrix
        The 2x2 ABCD matrix for refraction at a dielectric interface.
    '''
    if callable(refractive_index_in) or callable(refractive_index_out):
        n_in, n_out = refractive_index_in, refractive_index_out

        def matrix(wavelength):
            n1 = n_in(wavelength) if callable(n_in) else n_in
            n2 = n_out(wavelength) if callable(n_out) else n_out

            return make_abcd_refractive_interface(n1, n2, radius=radius, xp=xp)(wavelength)

        return RayTransferMatrix(matrix)

    if xp is None:
        xp = infer_xp(refractive_index_in, refractive_index_out, radius)

    n1 = refractive_index_in
    n2 = refractive_index_out

    if radius is None:
        return RayTransferMatrix(xp.asarray([[1.0, 0.0], [0.0, n1 / n2]]))

    return RayTransferMatrix(xp.asarray([[1.0, 0.0], [(n1 - n2) / (radius * n2), n1 / n2]]))
