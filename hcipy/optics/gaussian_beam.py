import math
from array_api_compat import is_array_api_obj
from .._math.backends import array_namespace
from ..field import Field
from .wavefront import Wavefront

class _ScalarNamespace:
    '''Array API-style namespace backed by the math module, used when all beam
    parameters are plain Python scalars. Its functions return Python floats.
    '''
    real = staticmethod(lambda x: x.real)
    imag = staticmethod(lambda x: x.imag)
    sqrt = staticmethod(math.sqrt)
    atan = staticmethod(math.atan)
    exp = staticmethod(math.exp)
    log = staticmethod(math.log)
    inf = math.inf

class GaussianBeam(object):
    '''An analytical description of a light beam with a Gaussian profile.

    The complex beam parameter q and the refractive index n of the medium
    are the primary parameters of the beam, from which all other quantities
    are derived. The beam is immutable.

    Parameters
    ----------
    q : scalar or Array
        The complex beam parameter of the Gaussian beam.
    wavelength : scalar or Array
        The vacuum wavelength of the light.
    n : scalar or Array
        The refractive index of the medium in which the beam propagates.
    '''
    def __init__(self, q, wavelength, n=1.0):
        self._q = q
        self._wavelength = wavelength
        self._n = n

        params = (q, wavelength, n)
        if any(is_array_api_obj(p) for p in params):
            self._xp = array_namespace(*params)
        else:
            self._xp = _ScalarNamespace

    @property
    def q(self):
        '''The complex beam parameter of the Gaussian beam.
        '''
        return self._q

    complex_beam_parameter = q

    @property
    def n(self):
        '''The refractive index of the medium in which the beam propagates.
        '''
        return self._n

    refractive_index = n

    @property
    def wavelength(self):
        '''The vacuum wavelength of the light.
        '''
        return self._wavelength

    @property
    def z(self):
        '''The current distance from the beam waist.
        '''
        return self._xp.real(self.q)

    @property
    def zR(self):  # noqa: N802
        '''The Rayleigh distance of the Gaussian beam.
        '''
        return self._xp.imag(self.q)

    rayleigh_distance = zR

    @property
    def w0(self):
        '''The beam waist of the Gaussian beam.
        '''
        return self._xp.sqrt(self.zR * self.wavelength / (math.pi * self.n))

    beam_waist = w0

    @property
    def theta(self):
        '''The beam divergence of the Gaussian beam in the medium.
        '''
        return self.wavelength / (math.pi * self.n * self.w0)

    beam_divergence = theta

    @property
    def R(self):  # noqa: N802
        '''The current radius of curvature of the Gaussian beam.
        '''
        epsilon = 1e-16
        if abs(self.z) < epsilon:
            return self._xp.inf
        else:
            return self.z * (1 + (self.zR / self.z)**2)

    radius_of_curvature = R

    @property
    def psi(self):
        '''The current Gouy phase of the Gaussian beam.
        '''
        return self._xp.atan(self.z / self.zR)

    gouy_phase = psi

    @property
    def w(self):
        '''The current beam radius of the Gaussian beam.
        '''
        return self.w0 * self._xp.sqrt(1 + (self.z / self.zR)**2)

    beam_radius = w

    @property
    def FWHM(self):  # noqa: N802
        '''The current FWHM of the Gaussian beam.
        '''
        return self.w * self._xp.sqrt(2 * self._xp.log(2.0))

    full_width_half_maximum = FWHM

    @property
    def k(self):
        '''The wavenumber of the Gaussian beam in the medium.
        '''
        return 2 * math.pi * self.n / self.wavelength

    wavenumber = k

    def propagate(self, matrix):
        '''Propagate the Gaussian beam through an ABCD (ray-transfer) matrix.

        The complex beam parameter of the beam is transformed according to the
        ABCD law

            q' = (A q + B) / (C q + D),

        and the refractive index as

            n' = n / det(M).

        A new GaussianBeam with the propagated parameters is returned.
        The beam itself is not modified.

        Parameters
        ----------
        matrix : Array or callable
            The 2x2 ABCD (ray-transfer) matrix of the optical system, in the
            physical-angle convention. If callable, it will be evaluated at the
            wavelength of the beam (in meters).

        Returns
        -------
        GaussianBeam
            The propagated Gaussian beam.

        Raises
        ------
        NotImplementedError
            If the ABCD matrix contains complex elements.
        ValueError
            If the matrix is not 2x2.
        '''
        if callable(matrix):
            matrix = matrix(self.wavelength)

        xp = array_namespace(matrix)

        if xp.isdtype(matrix.dtype, 'complex floating'):
            raise NotImplementedError('Propagation through systems with complex ABCD matrices (e.g. Gaussian apertures) is not implemented.')

        if matrix.shape != (2, 2):
            raise ValueError('The ABCD matrix should be a 2x2 array.')

        A, B, C, D = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        q = (A * self.q + B) / (C * self.q + D)
        n = self.n / (A * D - B * C)

        return GaussianBeam(q, self.wavelength, n)

    def evaluate(self, grid):
        '''Evaluate the wavefront of the Gaussian beam at the current position on
        the given grid.

        Parameters
        ----------
        grid : Grid
            The grid on which to calculate the wavefront for the Gaussian beam.

        Returns
        -------
        Wavefront
            The evaluated wavefront of the Gaussian beam.
        '''
        if grid.is_('cartesian'):
            r2 = grid.x**2 + grid.y**2
        else:
            r2 = grid.as_('polar').r**2

        xp = array_namespace(r2)

        # Note: this can be computed faster if the grids are separated and Cartesian,
        # but we can optimize when/if needed.
        K1 = self.w0 / self.w
        K2 = xp.exp(-r2 / self.w**2)
        K3 = xp.exp(-1j * (self.k * self.z + self.k * r2 / (2 * self.R) - self.psi))

        return Wavefront(Field(K1 * K2 * K3, grid), self.wavelength)

    __call__ = evaluate
