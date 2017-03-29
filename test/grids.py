from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

coords_1 = RegularCoords(0.1, [51, 51])

r = np.logspace(0, 2, 101)
theta = np.linspace(-np.pi, np.pi, 21)
coords_2 = SeparatedCoords((r, theta))

grid_1 = CartesianGrid(coords_1)
plt.plot(grid_1.x, grid_1.y, '+')
plt.show()

grid_2 = PolarGrid(coords_2)
print(grid_2.r)

grid_3 = grid_2.as_('cartesian')
print(grid_3)

plt.plot(grid_3.x, grid_3.y, '+')
plt.show()
