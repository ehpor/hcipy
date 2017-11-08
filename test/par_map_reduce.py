from hcipy import *
import numpy as np
import time

def f(x):
	return x

a = np.arange(10)+1

b = par_map_reduce(f, a, f_reduce=reduce_multiply, use_progressbar=True)
print(b)