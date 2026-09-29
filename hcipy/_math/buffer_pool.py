import threading
from contextlib import contextmanager

import numpy as np


class BufferPool:
    '''A thread-safe pool of reusable, flat NumPy buffers.

    Buffers are stored as flat (1-D, C-contiguous) arrays and bucketed by
    ``(dtype, number of elements)``. Acquiring a buffer of a given shape
    returns a view with that shape; releasing it returns the underlying flat
    buffer to the pool so that it can be handed out again. If no buffer of the
    requested size is available, a new one is allocated from scratch.

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
        size = int(np.prod(shape)) if shape else 1
        key = (dtype.str, size)

        with self._lock:
            bucket = self._free.get(key)
            flat = bucket.pop() if bucket else np.empty(size, dtype)

        return flat.reshape(shape)

    def _release(self, array):
        '''Return `array` to the pool.

        The underlying flat buffer is pooled, so passing a view (such as the
        result of :meth:`_acquire`) is allowed. The array must not be used
        afterwards.
        '''
        if not isinstance(array, np.ndarray):
            raise TypeError('Can only release numpy arrays to the buffer pool.')

        flat = array
        while flat.base is not None:
            flat = flat.base

        if flat.ndim != 1:
            flat = flat.reshape(-1)

        key = (flat.dtype.str, flat.size)
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
    '''Context manager yielding a temporary buffer from the process-wide pool.

    See :meth:`BufferPool.temp`.
    '''
    return _pool.temp(shape, dtype)


def clear():
    '''Drop all buffers held by the process-wide pool.'''
    return _pool.clear()
