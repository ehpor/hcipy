from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

grid = make_pupil_grid(512)

a = Field(np.random.randn(grid.size), grid)
mask = circular_aperture(1)(grid)
a[~mask.astype('bool')] = None

imshow_field(a)
plt.show()