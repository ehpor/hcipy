import numpy as np

def svd(M, num_modes=None):
    if num_modes is None:
        from numpy.linalg import svd
        return svd(M, full_matrices=False)
    else:
        from scipy.sparse.linalg import svds
        return svds(M, int(num_modes))

# Incremental SVD planned