import threading
from contextlib import contextmanager

import numpy as np


def _round_up_to_power_of_two(n):
    '''Return the smallest power of two greater than or equal to `n`.'''
    if n <= 1:
        return n
    return 1 << (n - 1).bit_length()


class BufferPool:
    '''A thread-safe pool of reusable, flat NumPy buffers.

    Buffers are stored as flat (1-D, C-contiguous) ``uint8`` byte arrays and
    bucketed by their number of bytes rounded up to the next power of two.
    Acquiring a buffer of a given shape returns a typed view with that shape;
    releasing it returns the underlying flat byte buffer to the pool so that it
    can be handed out again. If no buffer of the requested size is available, a
    new one is allocated from scratch.

    Rounding the bucket size up to a power of two increases reuse between
    differently-sized requests, at the cost of a memory footprint of at most
    twice the requested size.

    Because buffers are bucketed by size in bytes, a released buffer can be
    reinterpreted by a later acquisition of a different dtype, as long as the
    total byte sizes fall into the same bucket. Data types containing Python
    objects are not supported, since reinterpreting uninitialized bytes as
    object pointers is unsafe.

    Acquired buffers have unspecified contents: a released buffer is handed
    back out as-is.

    A released buffer may still be referenced by the caller; the pool only
    promises that it can be handed out again. Buffers must therefore not be
    used after they have been released.

    This module exposes a single process-wide pool through the module-level
    functions :func:`temp` and :func:`clear`.
    '''
    def __init__(self):
        self._lock = threading.Lock()
        self._free = {}

    def _acquire(self, shape, dtype):
        '''Return a buffer with the requested shape and dtype.

        Parameters
        ----------
        shape : int or tuple of int
            The shape of the returned array. An integer is interpreted as a
            1-D shape.
        dtype : dtype
            The data type of the returned array.

        Returns
        -------
        ndarray
            A buffer with the requested shape and dtype. Its contents are
            unspecified.
        '''
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        else:
            shape = tuple(int(s) for s in shape)

        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise TypeError('The buffer pool does not support object dtypes.')

        size = int(np.prod(shape)) if shape else 1
        nbytes = size * dtype.itemsize
        bucket_size = _round_up_to_power_of_two(nbytes)
        if bucket_size % dtype.itemsize:
            # Non-power-of-two itemsizes cannot be viewed from a power-of-two
            # byte buffer; fall back to the exact size.
            bucket_size = nbytes

        with self._lock:
            bucket = self._free.get(bucket_size)
            flat = bucket.pop() if bucket else np.empty(bucket_size, dtype=np.uint8)

        return flat.view(dtype)[:size].reshape(shape)

    def _release(self, array):
        '''Return `array` to the pool.

        The underlying flat byte buffer is pooled, so passing a view (such as
        the result of :meth:`_acquire`) is allowed. The array must not be used
        afterwards.
        '''
        if not isinstance(array, np.ndarray):
            raise TypeError('Can only release numpy arrays to the buffer pool.')
        if array.dtype.hasobject:
            raise TypeError('The buffer pool does not support object dtypes.')

        flat = array
        while flat.base is not None:
            flat = flat.base

        if flat.dtype != np.uint8:
            flat = np.ascontiguousarray(flat).view(np.uint8)

        if flat.ndim != 1:
            flat = flat.reshape(-1)

        key = flat.size
        with self._lock:
            self._free.setdefault(key, []).append(flat)

    @contextmanager
    def temp(self, shape, dtype):
        '''Context manager yielding a temporary buffer with the given shape.

        The buffer is returned to the pool when the context exits, also when
        an exception is raised.

        Parameters
        ----------
        shape : int or tuple of int
            The shape of the yielded array.
        dtype : dtype
            The data type of the yielded array.

        Yields
        ------
        ndarray
            A buffer with the requested shape and dtype.
        '''
        array = self._acquire(shape, dtype)
        try:
            yield array
        finally:
            self._release(array)

    def clear(self):
        '''Drop all buffers currently held by the pool.'''
        with self._lock:
            self._free.clear()


_pool = BufferPool()


def temp(shape, dtype):
    '''Context manager yielding a temporary buffer with the given shape.

    The buffer is returned to the pool when the context exits, also when
    an exception is raised.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the yielded array.
    dtype : dtype
        The data type of the yielded array.

    Yields
    ------
    ndarray
        A buffer with the requested shape and dtype.
    '''
    return _pool.temp(shape, dtype)
