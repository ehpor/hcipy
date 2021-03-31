import matplotlib
matplotlib.use('Agg')

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
import shutil

def test_animation():
	grid = make_pupil_grid(256)

	mw_frame = FrameWriter('test_frames/')
	mw_gif = GifWriter('test.gif')
	mw_mp4 = FFMpegWriter('test.mp4')

	mws = [mw_frame, mw_gif, mw_mp4]

	for i in range(25):
		field = Field(np.random.randn(grid.size), grid)

		plt.clf()
		imshow_field(field)

		for mw in mws:
			mw.add_frame()

	for mw in mws:
		mw.close()

		pytest.raises(RuntimeError, mw.add_frame)

	assert os.path.isdir('test_frames')
	shutil.rmtree('test_frames')

	assert os.path.isfile('test.gif')
	assert not os.path.exists('test.gif.frames')
	os.remove('test.gif')

	assert os.path.isfile('test.mp4')
	os.remove('test.mp4')

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

def test_imshow_util():
	pupil_grid = make_pupil_grid(128)
	focal_grid = make_focal_grid(4, 16)

	aperture = make_magellan_aperture(True)(pupil_grid)
	prop = FraunhoferPropagator(pupil_grid, focal_grid)

	wf = Wavefront(aperture)
	wf.electric_field *= np.exp(0.1j * zernike(6, 2, radial_cutoff=False)(pupil_grid))

	imshow_pupil_phase(wf, remove_piston=True, crosshairs=True, title='phase')
	plt.clf()

	img = prop(wf)

	imshow_psf(img, colorbar_orientation='vertical', normalization='peak', crosshairs=True, title='psf')
	plt.clf()

	imshow_psf(img, scale='linear', colorbar_orientation='vertical', normalization='peak', crosshairs=True, title='psf')
	plt.clf()
