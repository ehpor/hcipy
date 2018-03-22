from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

grid = make_pupil_grid(1024)
f = np.cos(grid.x * 100) * np.sin(grid.y * 50)
f = Field(f, grid)
'''
imshow_field(f)
plt.show()
'''
interpolator = make_linear_interpolator_separated(f)
grid2 = make_pupil_grid(2048, 3)

g = interpolator(grid2)
imshow_field(g, vmin=3*f.min(), vmax=3*f.max())
plt.show()
