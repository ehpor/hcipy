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

    for i in range(5):
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
def test_frame_writer(style, tmpdir):
    dirname = os.path.join(tmpdir, 'test_frames/')
    mw = FrameWriter(dirname)

    check_animation(mw, style=style)

    assert os.path.isdir(dirname)

@pytest.mark.parametrize('style', ['mpl', 'fig', 'img'])
def test_gif_writer(style, tmpdir):
    fname = os.path.join(tmpdir, 'test.gif')
    mw = GifWriter(fname)

    check_animation(mw, style=style)

    assert os.path.isfile(fname)

@pytest.mark.parametrize('style', ['mpl', 'fig', 'img'])
@pytest.mark.skipif(not is_ffmpeg_installed(), reason='FFMpeg is not installed.')
def test_ffmpeg_writer(style, tmpdir):
    fname = os.path.join(tmpdir, 'test.mp4')
    mw = FFMpegWriter(fname)

    check_animation(mw, style=style)

    assert mw._repr_html_()

    assert os.path.isfile(fname)

grids = [
    pytest.param(make_pupil_grid(256), id='regularly_spaced_grid'),
    pytest.param(CartesianGrid(SeparatedCoords((np.linspace(-0.5, 0.5, 257), np.linspace(-0.5, 0.5, 257)))), id='separated_grid')
]

@pytest.mark.parametrize('grid', grids)
def test_imshow_field(grid):
    field = Field(np.random.randn(grid.size), grid)

    imshow_field(field)
    plt.draw()
    plt.clf()

    imshow_field(field, grid_units=2)
    plt.draw()
    plt.clf()

    mask = make_circular_aperture(1)(grid)

    imshow_field(field, mask=mask)
    plt.draw()
    plt.clf()

    field = Field(np.random.randn(grid.size) + 1j * np.random.randn(grid.size), grid)

    imshow_field(field)
    plt.draw()
    plt.clf()

def test_imsave_field(tmpdir):
    grid = make_pupil_grid(256)

    field = Field(np.random.randn(grid.size), grid)

    fname = os.path.join(tmpdir, 'field.png')
    imsave_field(fname, field)
    assert os.path.isfile(fname)

def test_contour_field():
    grid = make_pupil_grid(256)

    field = Field(np.random.randn(grid.size), grid)

    contour_field(field)
    plt.draw()
    plt.clf()

    contourf_field(field)
    plt.draw()
    plt.clf()

def test_imshow_util():
    pupil_grid = make_pupil_grid(128)
    focal_grid = make_focal_grid(4, 16)

    aperture = make_magellan_aperture(True)(pupil_grid)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    wf = Wavefront(aperture)
    wf.electric_field *= np.exp(0.1j * zernike(6, 2, radial_cutoff=False)(pupil_grid))

    imshow_pupil_phase(wf, remove_piston=True, crosshairs=True, title='phase')
    plt.draw()
    plt.clf()

    img = prop(wf)

    imshow_psf(img, colorbar_orientation='vertical', normalization='peak', crosshairs=True, title='psf')
    plt.draw()
    plt.clf()

    imshow_psf(img, scale='linear', colorbar_orientation='vertical', normalization='peak', crosshairs=True, title='psf')
    plt.draw()
    plt.clf()
