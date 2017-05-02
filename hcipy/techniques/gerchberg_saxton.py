# -*- coding: utf-8 -*-

## Gerchberg-Saxton algorithm to retrieve the phase of the
## electric field at the focal plane given its intensity

# Input: x_n, X_k, e
# x_n - Real-space magnitude
# X_k - Fourier magnitude
# e - Error threshold

# Output: z_n - a vector that conforms with both magnitude
# constraints, i.e. z_n = x_n, and Z_k = X_k,
# where Z_k is the DFT of z_n

import numpy as np

def gerchberg_saxton(x_n, X_k, e):
    
    phi_n = np.zeros(np.shape(x_n))
    z0_n = np.abs(x_n) * np.exp(phi_n)
    
    zi_n = z0_n
    Zi_k = np.fft.fft(zi_n)
    
    while np.sum((np.abs(Zi_k) - np.abs(X_k)) ** 2) <= e:
        
        Zi_k = np.fft.fft(zi_n)
        Zii_k = np.dot(np.abs(X_k), Zi_k / np.abs(Zi_k))
        zii_n = np.fft.ifft(Zii_k)
        zi_n = np.dot(np.abs(x_n), zii_n / np.abs(zii_n))
    
    return zi_n