from array_api_compat import is_numpy_namespace, is_torch_namespace, is_jax_namespace, is_cupy_namespace
import copy

class RandomState:
    '''A class that provides a consistent API for random number generation across different array API namespaces.

    Parameters
    ----------
    xp : object
        The array API namespace (e.g., numpy, torch, jax) to use for random number generation.
    seed : int, optional
        The seed for the random number generator. If ``None``, a random seed will be used, by default None.

    Raises
    ------
    ValueError
        If an unsupported array API namespace is provided.
    '''
    def __init__(self, xp, seed=None):
        self.xp = xp
        self._jax_random = None

        if is_numpy_namespace(xp) or is_cupy_namespace(xp):
            self._rng = xp.random.default_rng(seed)
        elif is_torch_namespace(xp):
            self._rng = xp.Generator()
            if seed is not None:
                self._rng.manual_seed(seed)
        elif is_jax_namespace(xp):
            from jax import random
            self._jax_random = random
            self._rng = random.PRNGKey(seed or 0)
        else:
            raise ValueError(f"Unsupported namespace: {xp}")

    def copy(self):
        '''Return a new RandomState object that is an independent copy of the current state.

        Returns
        -------
        RandomState
            A new RandomState object with the same internal state as the current object.
        '''
        new_rng = RandomState.__new__(RandomState)
        new_rng.xp = self.xp
        new_rng._jax_random = self._jax_random

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            new_rng._rng = copy.deepcopy(self._rng)
        elif is_torch_namespace(self.xp):
            new_rng._rng = self.xp.Generator()
            new_rng._rng.set_state(self.xp.get_rng_state())
        elif is_jax_namespace(self.xp):
            new_rng._rng = self._rng  # Immutable, safe to share reference

        return new_rng

    def normal(self, mean=0.0, std=1.0, *, size=None):
        """Generate random samples from a normal (Gaussian) distribution.

        Parameters
        ----------
        mean : float, optional
            Mean ("center") of the distribution, by default 0.0.
        std : float, optional
            Standard deviation (spread or "width") of the distribution, by default 1.0.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.

        Returns
        -------
        array
            An array of specified shape filled with random samples from the
            normal distribution.
        """
        if size is None:
            size = ()

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            return self._rng.normal(mean, std, size)
        elif is_torch_namespace(self.xp):
            return self.xp.randn(*size, generator=self._rng) * std + mean
        elif is_jax_namespace(self.xp):
            self._rng, subkey = self._jax_random.split(self._rng)
            return self._jax_random.normal(subkey, size) * std + mean

    def poisson(self, lam=1.0, *, size=None):
        """Generate random samples from a Poisson distribution.

        Parameters
        ----------
        lam : float, optional
            Rate parameter ("lambda") of the distribution, by default 1.0.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.

        Returns
        -------
        array
            An array of specified shape filled with random samples from the
            Poisson distribution.
        """
        if size is None:
            size = ()

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            return self._rng.poisson(lam, size)
        elif is_torch_namespace(self.xp):
            return self.xp.poisson(lam, size=size, generator=self._rng)
        elif is_jax_namespace(self.xp):
            self._rng, subkey = self._jax_random.split(self._rng)
            return self._jax_random.poisson(subkey, lam, size)

    def gamma(self, scale=1.0, shape_param=1.0, *, size=None):
        """Generate random samples from a Gamma distribution.

        Parameters
        ----------
        scale : float, optional
            The scale parameter (beta or theta, inverse of rate) of the distribution, by default 1.0.
        shape_param : float, optional
            The shape parameter (k or alpha) of the distribution, by default 1.0.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.

        Returns
        -------
        array
            An array of specified shape filled with random samples from the
            Gamma distribution.
        """
        if size is None:
            size = ()

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            return self._rng.gamma(shape_param, scale, size)
        elif is_torch_namespace(self.xp):
            return self.xp.gamma(shape_param, (1.0 / scale), size=size, generator=self._rng)
        elif is_jax_namespace(self.xp):
            self._rng, subkey = self._jax_random.split(self._rng)
            return self._jax_random.gamma(subkey, shape_param, size) * scale
