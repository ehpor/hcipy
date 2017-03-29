from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

pupil_grid = make_pupil_grid(512)

aperture = circular_aperture(1)

A = aperture(pupil_grid)
imshow_field(A)
plt.show()

weird_aperture = regular_polygon_aperture(5, 1)

imshow_field(weird_aperture(pupil_grid))
plt.show()