import functools

from .backend import _functions, _function_wrappers, _function_aliases, make_translator, call
import numpy as np

def tensorflow_to_numpy(x):
    return x.numpy()

def tensorflow_pad_wrap(tf_pad):
    def numpy_like(array, pad_width, mode='constant', constant_values=0):
        if mode != 'constant':
            raise NotImplementedError

        try:
            if len(pad_width) == 1:
                pad_width = pad_width * len(array.shape)
        except TypeError:
            pad_width = ((pad_width, pad_width),) * len(array.shape)

        return tf_pad(array, pad_width, mode='CONSTANT', constant_values=constant_values)

    return numpy_like

def tensorflow_where_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(condition, x=None, y=None, **kwargs):
        return tuple(transpose(fn(condition, x, y, **kwargs)))

    return numpy_like

def tensorflow_split_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(ary, indices_or_sections, axis=0, **kwargs):
        if isinstance(indices_or_sections, int):
            return fn(ary, indices_or_sections, axis=axis, **kwargs)
        else:
            diff = call('diff', indices_or_sections, prepend=0, append=ary.shape[axis], like='numpy')
            diff = list(diff)

            return fn(ary, diff, axis=axis)

    return numpy_like

def tensorflow_dot(a, b):
	if np.isscalar(a) or np.isscalar(b):
		return a * b

	if a.ndim == 1 and b.ndim == 1:
		return call('sum', a * b)

	if a.ndim > 1 and b.ndim == 1:
		return call('linalg.matvec', a, b)

	return call('matmul', a, b)

_functions['tensorflow']['to_numpy'] = tensorflow_to_numpy
_functions['tensorflow']['dot'] = tensorflow_dot

_function_aliases['tensorflow', 'fft.fft2'] = 'signal.fft2d'
_function_aliases['tensorflow', 'log'] = 'math.log'
_function_aliases['tensorflow', 'conj'] = 'math.conj'
_function_aliases['tensorflow', 'real'] = 'math.real'
_function_aliases['tensorflow', 'imag'] = 'math.imag'
_function_aliases['tensorflow', 'power'] = 'math.power'
_function_aliases['tensorflow', 'count_nonzero'] = 'math.count_nonzero'
_function_aliases['tensorflow', 'diag'] = 'linalg.diag'
_function_aliases['tensorflow', 'trace'] = 'linalg.trace'
_function_aliases['tensorflow', 'tril'] = 'linalg.tril'
_function_aliases['tensorflow', 'triu'] = 'linalg.triu'
_function_aliases['tensorflow', 'sum'] = 'reduce_sum'
_function_aliases['tensorflow', 'min'] = 'reduce_min'
_function_aliases['tensorflow', 'max'] = 'reduce_max'
_function_aliases['tensorflow', 'mean'] = 'reduce_mean'
_function_aliases['tensorflow', 'prod'] = 'reduce_prod'
_function_aliases['tensorflow', 'concatenate'] = 'concat'
_function_aliases['tensorflow', 'clip'] = 'clip_by_value'
_function_aliases['tensorflow', 'arange'] = 'range'
_function_aliases['tensorflow', 'tril'] = 'band_part'
_function_aliases['tensorflow', 'triu'] = 'band_part'
_function_aliases['tensorflow', 'diag'] = 'tensor_diag'
_function_aliases['tensorflow', 'array'] = 'convert_to_tensor'
_function_aliases['tensorflow', 'astype'] = 'cast'
_function_aliases['tensorflow', 'power'] = 'pow'
_function_aliases['tensorflow', 'take'] = 'gather'

#_function_wrappers['tensorflow', 'linalg.svd'] = svd_sUV_to_UsVH_wrapper
#_function_wrappers['tensorflow', 'linalg.qr'] = qr_allow_fat
#_function_wrappers['tensorflow', 'linalg.solve'] = binary_allow_1d_rhs_wrap
#_function_wrappers['tensorflow', 'matmul'] = binary_allow_1d_rhs_wrap
#_function_wrappers['tensorflow', 'tril'] = tril_to_band_part
#_function_wrappers['tensorflow', 'triu'] = triu_to_band_part
_function_wrappers['tensorflow', 'pad'] = tensorflow_pad_wrap
_function_wrappers['tensorflow', 'where'] = tensorflow_where_wrap
_function_wrappers['tensorflow', 'split'] = tensorflow_split_wrap
_function_wrappers['tensorflow.random', 'uniform'] = make_translator(
    [
        ('low', ('minval', 0.0)),
        ('high', ('maxval', 1.0)),
        ('size', ('shape', ())),
    ]
)
_function_wrappers['tensorflow.random', 'normal'] = make_translator(
    [
        ('loc', ('mean', 0.0)),
        ('scale', ('stddev', 1.0)),
        ('size', ('shape', ())),
    ]
)
_function_wrappers['tensorflow', 'clip'] = make_translator(
    [
        ('a', ('t', 0.0)),
        ('a_min', ('clip_value_min',)),
        ('a_max', ('clip_value_max',)),
    ]
)
