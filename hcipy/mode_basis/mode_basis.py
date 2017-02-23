import numpy as np
 
class ModeBasis(list):
	@property
	def transformation_matrix(self):
		return np.column_stack(self)
	
	# TODO: currently only works on scalar fields
	@property
	def orthogonalized(self):
		q, r = np.linalg.qr(self.transformation_matrix)
		return ModeBasis((q[:,i] for i in range(q.shape[1])))