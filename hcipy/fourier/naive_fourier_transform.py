import numpy as np
from .fourier_transform import FourierTransform

class NaiveFourierTransform(FourierTransform):
    def __init__(self, input_grid, output_grid):
        self.input_grid = input_grid
        self.output_grid = output_grid
    
    def forward(self, field):
        T = self.get_transformation_matrix_forward()
        res = T.dot(field.ravel())

        from ..field import Field
        return Field(res, self.output_grid)
    
    def backward(self, field):
        T = self.get_transformation_matrix_backward()
        res = T.dot(field.ravel())

        from ..field import Field
        return Field(res, self.input_grid)