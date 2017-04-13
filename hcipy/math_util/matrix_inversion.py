import numpy as np

def inverse_truncated_modal(M, num_modes, svd=None):
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M, num_modes)

	U, S, Vt = svd.svd
	return (Vt.T / S).dot(U.T)

def inverse_truncated(M, rcond=1e-15, svd=None):
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M)

	U, S, Vt = svd.svd
	S_inv = np.array([1/s if abs(s) > (rocond * S.max()) else 0 for s in S])

	return (Vt.T * S_inv).dot(U.T)

def inverse_tikhonov(M, rcond=1e-15, svd=None):
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M)
	
	U, S, Vt = svd.svd
	S_inv = S / (S**2 + (rcond * S.max())**2)

	return (Vt.T * S_inv).dot(U.T)