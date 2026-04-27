from array_api_compat import is_numpy_namespace, is_torch_namespace, is_jax_namespace, is_cupy_namespace
import copy
import math


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


class RandomGenerator:
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
        '''Return a new RandomGenerator object that is an independent copy of the current state.

        Returns
        -------
        RandomGenerator
            A new RandomGenerator object with the same internal state as the current object.
        '''
        new_rng = RandomGenerator.__new__(RandomGenerator)
        new_rng.xp = self.xp
        new_rng._jax_random = self._jax_random

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            new_rng._rng = copy.deepcopy(self._rng)
        elif is_torch_namespace(self.xp):
            new_rng._rng = self.xp.Generator()
            new_rng._rng.set_state(self._rng.get_state())
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
            size = (1,)
        elif not isinstance(size, tuple):
            size = (size,)

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
            size = (1,)
        elif not isinstance(size, tuple):
            size = (size,)

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            return self._rng.poisson(lam, size)
        elif is_torch_namespace(self.xp):
            lam = self.xp.ones(size=size) * lam
            return self.xp.poisson(lam, generator=self._rng)
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
            size = (1,)
        elif not isinstance(size, tuple):
            size = (size,)

        if is_numpy_namespace(self.xp) or is_cupy_namespace(self.xp):
            return self._rng.gamma(shape_param, scale, size)
        elif is_torch_namespace(self.xp):
            return _torch_gamma(scale, shape_param, size=size, generator=self._rng)
        elif is_jax_namespace(self.xp):
            self._rng, subkey = self._jax_random.split(self._rng)
            return self._jax_random.gamma(subkey, shape_param, size) * scale
