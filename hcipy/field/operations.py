from .field import Field
import numpy as np

def field_inverse_tikhonov(f, rcond=1e-15):
	from ..math_util import inverse_tikhonov

	res = []
	for i in range(f.grid.size):
		res.append(inverse_tikhonov(f[...,i], rcond))
	return Field(np.moveaxis(res, 0, -1), f.grid)

def make_field_operation(op):
	pass