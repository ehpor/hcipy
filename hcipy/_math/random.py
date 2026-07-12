from __future__ import annotations

from array_api_compat import is_numpy_namespace, is_torch_namespace, is_jax_namespace, is_cupy_namespace
from .typing import Array, ArrayNamespace, ArrayT
import copy
import math
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def make_random_generator(xp: ArrayNamespace[ArrayT], seed: int | None = None) -> RandomGenerator:
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


def _torch_gamma(scale: float = 1.0, shape: float = 1.0, size: tuple[int, ...] | None = None, generator: torch.Generator | None = None) -> Array:
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


def _normalize_size(size: int | tuple[int, ...] | None) -> tuple[int, ...]:
    if size is None:
        return (1,)
    elif not isinstance(size, tuple):
        return (size,)
    return size


class RandomGenerator:
    def copy(self) -> RandomGenerator:
        '''Return a new RandomGenerator object that is an independent copy of the current state.

        Returns
        -------
        RandomGenerator
            A new RandomGenerator object with the same internal state as the current object.
        '''
        raise NotImplementedError()

    def normal(self, mean: float = 0.0, std: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
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

    def poisson(self, lam: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
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

    def gamma(self, scale: float = 1.0, shape_param: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
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


class RandomGeneratorNumpy(RandomGenerator):
    def __init__(self, seed: int | None = None) -> None:
        import numpy

        self._rng = numpy.random.default_rng(seed)

    def copy(self) -> RandomGenerator:
        res = RandomGeneratorNumpy()
        res._rng = copy.deepcopy(self._rng)

        return res

    def normal(self, mean: float = 0.0, std: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        return cast(Array, self._rng.normal(mean, std, size))

    def poisson(self, lam: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        return cast(Array, self._rng.poisson(lam, size))

    def gamma(self, scale: float = 1.0, shape_param: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        return cast(Array, self._rng.gamma(shape_param, scale, size))


class RandomGeneratorCupy(RandomGeneratorNumpy):
    def __init__(self, seed: int | None = None) -> None:
        import cupy
        self._rng = cupy.random.default_rng(seed)


class RandomGeneratorTorch(RandomGenerator):
    def __init__(self, seed: int | None = None) -> None:
        import torch
        self._xp = torch

        self._rng = self._xp.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def copy(self) -> RandomGenerator:
        res = RandomGeneratorTorch()
        res._rng = self._xp.Generator()
        res._rng.set_state(self._rng.get_state())

        return res

    def normal(self, mean: float = 0.0, std: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        return self._xp.randn(*size, generator=self._rng) * std + mean

    def poisson(self, lam: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        lam_tensor = self._xp.ones(size=size) * lam
        return self._xp.poisson(lam_tensor, generator=self._rng)

    def gamma(self, scale: float = 1.0, shape_param: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        return _torch_gamma(scale, shape_param, size=size, generator=self._rng)


class RandomGeneratorJax(RandomGenerator):
    def __init__(self, seed: int | None = None) -> None:
        from jax import random
        import jax.numpy

        self._xp = jax.numpy
        self._jax_random = random

        self._rng = random.PRNGKey(seed or 0)

    def copy(self) -> RandomGenerator:
        res = RandomGeneratorJax()
        res._jax_random = self._jax_random
        res._rng = self._rng

        return res

    def normal(self, mean: float = 0.0, std: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return cast(Array, self._jax_random.normal(subkey, size) * std + mean)

    def poisson(self, lam: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return cast(Array, self._jax_random.poisson(subkey, lam, shape=size))

    def gamma(self, scale: float = 1.0, shape_param: float = 1.0, *, size: int | tuple[int, ...] | None = None) -> Array:
        size = _normalize_size(size)
        self._rng, subkey = self._jax_random.split(self._rng)
        return cast(Array, self._jax_random.gamma(subkey, shape_param, shape=size) * scale)
