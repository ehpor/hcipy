import numpy as np
 
class ModeBasis(list):
	@property
	def transformation_matrix(self):
		return np.stack(self, axis=-1)
	
	# TODO: currently only works on scalar fields
	@property
	def orthogonalized(self):
		q, r = np.linalg.qr(self.transformation_matrix)
		if hasattr(self[0], 'grid'):
			from ..field import Field
			grid = self[0].grid
			return ModeBasis((Field(q[:,i], grid) for i in range(q.shape[1])))
		else:
			return ModeBasis((q[:,i] for i in range(q.shape[1])))