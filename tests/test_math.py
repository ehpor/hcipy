from hcipy.math import numpy as np
from hcipy import *
import time

a = np.eye(1024, dtype='complex64', like='cupy')

print('Dot product:')
b = np.dot(a, a)
print(np.to_numpy(b).dtype)

N = 10000
start = time.perf_counter()
for i in range(N):
    b = np.dot(a, a)
np.to_numpy(b)
end = time.perf_counter()
print((end - start) / N * 1e6, 'us per iteration')

print('FFT:')
b = np.fft.fft2(a)
print(np.to_numpy(b).dtype)

N = 10000
start = time.perf_counter()
for i in range(N):
    b = np.fft.fft2(a)
np.to_numpy(b)
end = time.perf_counter()
print((end - start) / N * 1e6, 'us per iteration')
