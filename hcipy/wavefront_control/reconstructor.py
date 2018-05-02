import enum

class CovarianceType(enum.Enum):
	CovNone = 0
	CovScalar = 1
	CovVector = 2
	CovMatrix = 3

class Reconstructor(object):
	def __init__(self, covariance_required=CovarianceType.CovNone):
		self.covariance_required = covariance_required
	
	def reconstruct(self, t, wavefront, covariances=None):
		raise NotImplementedError()