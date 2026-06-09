from array_api_compat import is_numpy_namespace, is_torch_namespace, is_jax_namespace, is_cupy_namespace
import copy
import math


def make_random_generator(xp, seed=None):
    '''Create a RandomGenerator instance for the given array API namespace.

    Parameters
    ----------
    xp : object
        The array API namespace (e.g., numpy, torch, jax, cupy).
    seed : int, optional
        The seed for the random number generator, by default None.

    Returns
    -------
    RandomGenerator
        An instance of the appropriate RandomGenerator subclass for the given namespace.
    '''
    if is_numpy_namespace(xp):
        return RandomGeneratorNumpy(seed)
    elif is_cupy_namespace(xp):
        return RandomGeneratorCupy(seed)
    elif is_torch_namespace(xp):
        return RandomGeneratorTorch(seed)
    elif is_jax_namespace(xp):
        return RandomGeneratorJax(seed)
    else:
        raise ValueError(f"Unsupported namespace: {xp}")


def _torch_gamma(scale=1.0, shape=1.0, size=None, generator=None):
    '''Sample from Gamma(shape, scale) distribution using pytorch.

    Pytorch does not support a generator argument for gamma, so we need to implement it.
    This implementation uses the Marsaglia-Tsang method.

    Parameters
    ----------
    scale : float
        The scale parameter of the Gamma distribution.
    shape : float
        The shape parameters of the Gamma distribution.
    size : tuple or None
        The size of the output tensor.
    generator : torch.Generator or None
        The random number generator to use.

    Returns
    -------
    torch.Tensor
        The Gamma-distributed generated samples.
    '''
    import torch

    if size is None:
        size = (1,)

    n = math.prod(size)

    # For shape < 1, use the boost: if X ~ Gamma(a+1), then X*U^(1/a) ~ Gamma(a)
    boost = shape < 1
    a = shape + 1 if boost else shape

    d = a - 1 / 3
    c = 1 / math.sqrt(9.0 * d)

    # Use batches, be conservative. Average acceptance rate is ~98% so a factor of 2x is appropriate.
    batch_size = max(n * 2, 2**16)

    samples = []
    num_samples = 0
    while num_samples < n:
        x = torch.randn(batch_size, generator=generator)
        v = (1 + c * x)**3

        u = torch.rand(batch_size, generator=generator)

        squeeze = u < 1 - 0.0331 * x**4
        log_check = torch.log(u) < 0.5 * x**2 + d * (1 - v + torch.log(v.clamp(min=1e-10)))

        accepted_mask = (v > 0) & (squeeze | log_check)
        samples.append((d * v)[accepted_mask])

        num_samples += torch.sum(accepted_mask)

    samples = torch.cat(samples)[:n]

    if boost:
        u = torch.rand(n, generator=generator)
        samples = samples * u ** (1 / shape)

    return (samples * scale).reshape(size)


def _normalize_size(size):
    if size is None:
        return (1,)
    elif not isinstance(size, tuple):
        return (size,)
    return size


class RandomGenerator:
    def __init__(self, xp):
        self._xp = xp

    def copy(self):
        '''Return a new RandomGenerator object that is an independent copy of the current state.

        Returns
        -------
        RandomGenerator
            A new RandomGenerator object with the same internal state as the current object.
        '''
        new_rng = self.__new__(type(self))
        new_rng._xp = self._xp

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def uniform(self, low=0.0, high=1.0, *, size=None):
        """Generate random samples from a uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution, by default 0.0.
        high : float, optional
            Upper bound of the distribution, by default 1.0.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.

        Returns
        -------
        array
            An array of specified shape filled with random samples from the
            uniform distribution.
        """
        raise NotImplementedError()

    def exponential(self, scale=1.0, *, size=None):
        """Generate random samples from an exponential distribution.

        Parameters
        ----------
        scale : float, optional
            The scale parameter (inverse of rate), by default 1.0.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.

        Returns
        -------
        array
            An array of specified shape filled with random samples from the
            exponential distribution.
        """
        raise NotImplementedError()

    def choice(self, a, *, size=None, replace=True, p=None):
        """Generate random samples from a given array.

        Parameters
        ----------
        a : array-like or int
            If an array, a random sample is generated from its elements.
            If an int, the random sample is generated from ``range(a)``.
        size : int or tuple of ints, optional
            Output shape. If ``None``, a single scalar value is returned, by default None.
        replace : bool, optional
            Whether to sample with replacement, by default True.
        p : array-like, optional
            The probabilities of each element in ``a``. If not given, the
            sample assumes a uniform distribution over all entries in ``a``.

        Returns
        -------
        array
            An array of specified shape filled with random samples from
            the given array.
        """
        raise NotImplementedError()


class RandomGeneratorNumpy(RandomGenerator):
    def __init__(self, seed=None):
        import numpy
        super().__init__(numpy)

        self._rng = self._xp.random.default_rng(seed)

    def copy(self):
        res = super().copy()
        res._rng = copy.deepcopy(self._rng)

        return res

    def normal(self, mean=0.0, std=1.0, *, size=None):
        size = _normalize_size(size)
        return self._rng.normal(mean, std, size)

    def poisson(self, lam=1.0, *, size=None):
        size = _normalize_size(size)
        return self._rng.poisson(lam, size)

    def gamma(self, scale=1.0, shape_param=1.0, *, size=None):
        size = _normalize_size(size)
        return self._rng.gamma(shape_param, scale, size)

    def uniform(self, low=0.0, high=1.0, *, size=None):
        size = _normalize_size(size)
        return self._rng.uniform(low, high, size)

    def exponential(self, scale=1.0, *, size=None):
        size = _normalize_size(size)
        return self._rng.exponential(scale, size)

    def choice(self, a, *, size=None, replace=True, p=None):
        size = _normalize_size(size)
        return self._rng.choice(a, size, replace=replace, p=p)


class RandomGeneratorCupy(RandomGeneratorNumpy):
    def __init__(self, seed=None):
        import cupy
        super().__init__(cupy)

        self._rng = self._xp.random.default_rng(seed)


class RandomGeneratorTorch(RandomGenerator):
    def __init__(self, seed=None):
        import torch
        super().__init__(torch)

        self._rng = self._xp.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def copy(self):
        res = super().copy()
        res._rng = self._xp.Generator()
        res._rng.set_state(self._rng.get_state())

        return res

    def normal(self, mean=0.0, std=1.0, *, size=None):
        size = _normalize_size(size)
        return self._xp.randn(*size, generator=self._rng) * std + mean

    def poisson(self, lam=1.0, *, size=None):
        size = _normalize_size(size)
        lam_tensor = self._xp.ones(size=size) * lam
        return self._xp.poisson(lam_tensor, generator=self._rng)

    def gamma(self, scale=1.0, shape_param=1.0, *, size=None):
        size = _normalize_size(size)
        return _torch_gamma(scale, shape_param, size=size, generator=self._rng)

    def uniform(self, low=0.0, high=1.0, *, size=None):
        size = _normalize_size(size)
        return self._xp.rand(*size, generator=self._rng) * (high - low) + low

    def exponential(self, scale=1.0, *, size=None):
        size = _normalize_size(size)
        return -scale * self._xp.log(self._xp.rand(*size, generator=self._rng))

    def choice(self, a, *, size=None, replace=True, p=None):
        import math as _math

        size = _normalize_size(size)
        n = _math.prod(size)

        if isinstance(a, int):
            population = None
            n_pop = a
        else:
            population = self._xp.asarray(a)
            n_pop = len(population)

        if p is None:
            indices = self._xp.randint(0, n_pop, size, generator=self._rng)
        else:
            p_tensor = self._xp.asarray(p, dtype=self._xp.float64)
            p_tensor = p_tensor / p_tensor.sum()
            indices = self._xp.multinomial(p_tensor, n, replacement=replace, generator=self._rng)
            indices = indices.reshape(size)

        if population is None:
            return indices
        return population[indices]


class RandomGeneratorJax(RandomGenerator):
    def __init__(self, seed=None):
        from jax import random
        import jax.numpy
        super().__init__(jax.numpy)
        self._jax_random = random

        self._rng = random.PRNGKey(seed or 0)

    def copy(self):
        res = super().copy()
        res._jax_random = self._jax_random
        res._rng = self._rng

        return res

    def normal(self, mean=0.0, std=1.0, *, size=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.normal(subkey, size) * std + mean

    def poisson(self, lam=1.0, *, size=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.poisson(subkey, lam, shape=size)

    def gamma(self, scale=1.0, shape_param=1.0, *, size=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.gamma(subkey, shape_param, shape=size) * scale

    def uniform(self, low=0.0, high=1.0, *, size=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.uniform(subkey, shape=size, minval=low, maxval=high)

    def exponential(self, scale=1.0, *, size=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.exponential(subkey, shape=size) * scale

    def choice(self, a, *, size=None, replace=True, p=None):
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return self._jax_random.choice(subkey, a, shape=size, replace=replace, p=p)

