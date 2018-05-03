from hcipy import *
import numpy as np

def test_par_map():
	a = np.random.randn(1024,2)
	b = a[:,0] * a[:,1]

	def f(a):
		return a[0] * a[1]
	
	c = par_map(f, a)

	assert np.allclose(b, c)

def test_par_map_reseed():
	def f(x):
		return np.random.randn(1)
	
	a = np.array(par_map(f, np.arange(100)))
	assert not np.allclose(a, a[0])

def test_par_map_reduce():
	a = np.random.randn(1024,2)
	b = np.sum(a[:,0] * a[:,1])

	def f(a):
		return a[0] * a[1]
	
	c = par_map_reduce(f, a)

	assert np.allclose(b, c)

	d = np.prod(a[:,0] + a[:,1])
	
	def g(a):
		return a[0] + a[1]
	
	e = par_map_reduce(f, a, reduce_multiply)

	assert np.allclose(d, e)