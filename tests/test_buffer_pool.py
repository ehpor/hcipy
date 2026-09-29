import threading
import time

import numpy as np
import pytest

from hcipy._math.buffer_pool import BufferPool, temp, clear


def flat_base(array):
    base = array
    while base.base is not None:
        base = base.base
    return base


def test_temp_shape_and_dtype():
    pool = BufferPool()

    with pool.temp((3, 4), np.float64) as array:
        assert array.shape == (3, 4)
        assert array.dtype == np.float64
        # The yielded array is a shaped view onto a flat pooled byte buffer.
        assert flat_base(array).ndim == 1
        assert flat_base(array).dtype == np.uint8
        assert flat_base(array).size == 12 * np.dtype(np.float64).itemsize


def test_temp_integer_shape_is_1d():
    pool = BufferPool()

    with pool.temp(5, np.complex128) as array:
        assert array.shape == (5,)
        assert array.dtype == np.complex128


def test_temp_reuses_buffer():
    pool = BufferPool()

    with pool.temp((2, 3), np.float64) as first:
        base = flat_base(first)

    with pool.temp((2, 3), np.float64) as second:
        assert flat_base(second) is base


def test_temp_does_not_reuse_different_size():
    pool = BufferPool()

    with pool.temp((2, 3), np.float64) as first:
        base = flat_base(first)

    with pool.temp((2, 4), np.float64) as second:
        assert flat_base(second) is not base


def test_temp_reuses_same_bytes_different_dtype():
    # Buckets are keyed by byte size, so dtypes can share buffers.
    pool = BufferPool()

    with pool.temp((2, 3), np.complex128) as first:
        base = flat_base(first)

    with pool.temp(12, np.float64) as second:
        assert flat_base(second) is base


def test_temp_does_not_reuse_different_byte_size():
    pool = BufferPool()

    with pool.temp((2, 3), np.float64) as first:
        base = flat_base(first)

    with pool.temp((2, 3), np.complex128) as second:
        # 48 bytes versus 96 bytes.
        assert flat_base(second) is not base


def test_temp_rejects_object_dtype():
    pool = BufferPool()

    with pytest.raises(TypeError):
        with pool.temp((3,), object):
            pass


def test_release_rejects_object_dtype():
    pool = BufferPool()

    with pytest.raises(TypeError):
        pool._release(np.empty(3, dtype=object))


def test_temp_reuses_equal_size_different_shape():
    # Buffers are bucketed by number of elements, not by original shape.
    pool = BufferPool()

    with pool.temp((2, 3), np.float64) as first:
        base = flat_base(first)

    with pool.temp(6, np.float64) as second:
        assert flat_base(second) is base


def test_temp_releases_on_exception():
    pool = BufferPool()

    with pytest.raises(RuntimeError):
        with pool.temp((4, 4), np.complex128) as array:
            base = flat_base(array)
            raise RuntimeError('boom')

    with pool.temp((4, 4), np.complex128) as again:
        assert flat_base(again) is base


def test_clear_drops_buffers():
    pool = BufferPool()

    with pool.temp((8,), np.float64) as first:
        base = flat_base(first)

    pool.clear()

    with pool.temp((8,), np.float64) as second:
        assert flat_base(second) is not base


def test_module_level_temp_and_clear():
    with temp((4, 2), np.complex128) as array:
        assert array.shape == (4, 2)
        base = flat_base(array)

    with temp((4, 2), np.complex128) as again:
        assert flat_base(again) is base

    clear()


def test_thread_safety_no_concurrent_sharing():
    pool = BufferPool()

    n_threads = 8
    iterations = 300
    shape = (64, 64)
    dtype = np.complex128

    in_use = set()
    state_lock = threading.Lock()
    errors = []
    start = threading.Barrier(n_threads)

    def worker(thread_id):
        try:
            start.wait()
            for _ in range(iterations):
                with pool.temp(shape, dtype) as array:
                    base = flat_base(array)

                    with state_lock:
                        if id(base) in in_use:
                            raise AssertionError('buffer handed out to two threads at once')
                        in_use.add(id(base))

                    # Write a thread-specific pattern and make sure no other
                    # thread clobbers it while we hold the buffer.
                    array.fill(thread_id)
                    time.sleep(0.0002)
                    if not np.all(array == thread_id):
                        raise AssertionError('buffer contents were modified while in use')

                    with state_lock:
                        in_use.discard(id(base))
        except Exception as error:  # pragma: no cover - only on failure
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert not in_use
