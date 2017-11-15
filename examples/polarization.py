from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

pupil_grid = make_pupil_grid(512)
focal_grid = make_focal_grid(pupil_grid, 8, 16)
aperture = circular_aperture(1)
'''
E = np.empty((pupil_grid.size, 3))
E = Field(E, pupil_grid)
E[:,0] = aperture(pupil_grid)

print('Everything is valid:', E.is_valid_field)

wf = Wavefront(E)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
'''
a = Field(np.random.randn(pupil_grid.size,3,3), pupil_grid)
b = Field(np.random.randn(pupil_grid.size,3,3), pupil_grid)

print(a.is_valid_field)

res = field_einsum('ii->i', a)
print(res.is_valid_field)