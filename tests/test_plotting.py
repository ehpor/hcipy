import matplotlib
matplotlib.use('Agg')

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
import shutil

def is_ffmpeg_installed():
    ffmpeg_path = Configuration().plotting.ffmpeg_path

    if ffmpeg_path is None:
        ffmpeg_path = 'ffmpeg'

    return shutil.which(ffmpeg_path) is not None

def check_animation(mw, style):
    grid = make_pupil_grid(256)

    for i in range(25):
        field = Field(np.random.randn(grid.size), grid)

        if style == 'mpl':
            plt.clf()
            imshow_field(field)

            mw.add_frame()
        elif style == 'fig':
            fig, ax = plt.subplots()
            imshow_field(field, ax=ax)

            mw.add_frame(fig=fig)
            plt.close(fig)
        elif style == 'img':
            mw.add_frame(data=field.shaped, cmap='RdBu')
        else:
            raise ValueError('Animation style not known.')

    mw.close()

    pytest.raises(RuntimeError, mw.add_frame)

@pytest.mark.parametrize('style', ['mpl', 'fig', 'img'])
def test_frame_writer(style):
    mw = FrameWriter('test_frames/')

    check_animation(mw, style=style)

    assert os.path.isdir('test_frames')
    shutil.rmtree('test_frames')

@pytest.mark.parametrize('style', ['mpl', 'fig', 'img'])
def test_gif_writer(style):
    mw = GifWriter('test.gif')

    check_animation(mw, style=style)

    assert os.path.isfile('test.gif')
    os.remove('test.gif')

@pytest.mark.parametrize('style', ['mpl', 'fig', 'img'])
@pytest.mark.skipif(not is_ffmpeg_installed(), reason='FFMpeg is not installed.')
def test_ffmpeg_writer(style):
    mw = FFMpegWriter('test.mp4')

    check_animation(mw, style=style)

    assert mw._repr_html_()

    assert os.path.isfile('test.mp4')
    os.remove('test.mp4')

def test_imshow_field():
    grid = make_pupil_grid(256)

    field = Field(np.random.randn(grid.size), grid)

    imshow_field(field)
    plt.clf()

    mask = make_circular_aperture(1)(grid)

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
