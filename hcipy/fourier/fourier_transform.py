from numpy import *

class FourierTransform(object):
    def forward(self, field):
        raise NotImplementedError()
    
    def backward(self, field):
        raise NotImplementedError()
    
    def get_transformation_matrix(self):
        coords_in = self.input_grid.as_('cartesian').coords
        coords_out = self.output_grid.as_('cartesian').coords

        A = np.exp(-1j * np.dot(coords_out, coords_in.T))
        A *= coords_in.weights

        return A