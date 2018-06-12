from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

grid = make_pupil_grid(512)

a = Field(np.random.randn(grid.size), grid)
a += (grid.x**2 + grid.y**2) * 100
mask = circular_aperture(1)(grid)

imshow_field(a, mask=mask, mask_color='k')
plt.colorbar()
plt.show()