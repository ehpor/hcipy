import matplotlib
matplotlib.use('Agg')

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest

def test_gif_writer():
	grid = make_pupil_grid(256)

	mw = GifWriter('test.gif')

	for i in range(25):
		field = Field(np.random.randn(grid.size), grid)

		plt.clf()
		imshow_field(field)
		
		mw.add_frame()
	
	mw.close()

	assert os.path.isfile('test.gif')
	assert not os.path.exists('test.gif.frames')

	pytest.raises(RuntimeError, mw.add_frame)

	os.remove('test.gif')

def test_imshow_field():
	grid = make_pupil_grid(256)

	field = Field(np.random.randn(grid.size), grid)

	imshow_field(field)
	plt.clf()

	mask = circular_aperture(1)(grid)

	imshow_field(field, mask=mask)
	plt.clf()

	field = Field(np.random.randn(grid.size) + 1j * np.random.randn(grid.size), grid)

	imshow_field(field)
	plt.clf()

def test_imsave_field():
	grid = make_pupil_grid(256)

	field = Field(np.random.randn(grid.size), grid)

	imsave_field('field.png', field)
	assert os.path.isfile('field.png')

	os.remove('field.png')

def test_contour_field():
	grid = make_pupil_grid(256)

	field = Field(np.random.randn(grid.size), grid)

	contour_field(field)
	plt.clf()

	contourf_field(field)
	plt.clf()
