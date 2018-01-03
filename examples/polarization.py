from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

print(HalfWavePlate(lambda wl: 0).get_instance(None, 3))

pupil_grid = make_pupil_grid(32)
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
a = Field(np.random.randn(3,3,pupil_grid.size), pupil_grid)
b = Field(np.random.randn(3,pupil_grid.size), pupil_grid)
c = np.random.randn(3,3)

f = FresnelPropagator(pupil_grid, 2)
print(f.forward(Wavefront(a)).electric_field.tensor_shape)

print(a.grid.size)
print(b.grid.size)

res = field_trace(a)
res2 = np.array([np.trace(a[...,i]) for i in range(pupil_grid.size)])

print(np.allclose(res, res2))