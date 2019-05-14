import numpy as np

def inverse_truncated_modal(M, num_modes, svd=None):
	'''Invert `M` with `num_modes` modes.

	Modes are given by a singular value decomposition of the input matrix.
	If `svd` is given, it is used instead. Modes are ordered in descending
	order by their singular values.

	Parameters
	----------
	M : ndarray
		The matrix to invert. This matrix must be two-dimensional.
	num_modes : integer
		The number of modes to use for the inversion. Higher-order modes
		are ignored.
	svd : SVD object
		A precalculated singular value decomposition object. This will
		be used if supplied, to avoid recalculation of an SVD.
	
	Returns
	-------
	ndarray
		The inverted matrix.
	'''
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M, num_modes)

	U, S, Vt = svd.svd

	U = U[:,:num_modes]
	S = S[:num_modes]
	Vt = Vt[:num_modes,:]

	return (Vt.T / S).dot(U.T)

def inverse_truncated(M, rcond=1e-15, svd=None):
	'''Invert `M` truncating the number of modes.

	Modes are given by a singular value decomposition of the input matrix.
	If `svd` is given, it is used instead. Modes are ordered in descending
	order by their singular values. All modes with a singular value lower than 
	`rcond` times the maximum singular value, will be ignored.

	Parameters
	----------
	M : ndarray
		The matrix to invert. This matrix must be two-dimensional.
	rcond : scalar
		The relative condition number of the highest-order mode that must 
		be used for inversion.
	svd : SVD object
		A precalculated singular value decomposition object. This will
		be used if supplied, to avoid recalculation of an SVD.
	
	Returns
	-------
	ndarray
		The inverted matrix.
	'''
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M)

	U, S, Vt = svd.svd
	S_inv = np.array([1/s if abs(s) > (rcond * S.max()) else 0 for s in S])

	return (Vt.T * S_inv).dot(U.T)

def inverse_tikhonov(M, rcond=1e-15, svd=None):
	'''Invert `M` using Tikhonov regularization.

	Tikhonov regularization will be an identity matrix, with strength 
	`rcond` * the maximum singular value of the matrix.

	Parameters
	----------
	M : ndarray
		The matrix to invert. This matrix must be two-dimensional.
	rcond : scalar
		The relative strength of the regularization. An identity matrix
		will be used for the regularization.
	svd : SVD object
		A precalculated singular value decomposition object. This will
		be used if supplied, to avoid recalculation of an SVD.
	
	Returns
	-------
	ndarray
		The inverted matrix.
	'''
	if svd is None:
		from .singular_value_decomposition import SVD
		svd = SVD(M)
	
	U, S, Vt = svd.svd
	S_inv = S / (S**2 + (rcond * S.max())**2)

	return (Vt.T * S_inv).dot(U.T)