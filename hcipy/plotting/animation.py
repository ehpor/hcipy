import os
import shutil
import base64
from subprocess import Popen, PIPE
import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from ..config import Configuration

def _data_to_img(data, cmap, copy=False):
    if cmap is None:
        if copy:
            return data.copy()

        return data

    try:
        cmap = mpl.colormaps.get_cmap(cmap)
    except AttributeError:
        # For Matplotlib <3.5.
        cmap = mpl.cm.get_cmap(cmap)

    return cmap(data, bytes=True)

class FrameWriter(object):
    '''A writer of frames from Matplotlib figures.

    This class writes out individual frames to the specified directory. No
    animation will be created for these frames.

    Parameters
    ----------
    path : string
        The directory to which to write all frames. This path will be
        created if it does not exist.
    framerate : integer
        Ignored, but provided for cohesion with other animation writers.
    filename : string
        The filename for each of the frames. This will be formatted using
        filename.format(frame_number). Default: '{:05d}.png'.
    '''
    def __init__(self, path, framerate=15, filename='{:05d}.png'):
        self.path = path
        self.filename = filename

        self.num_frames = 0

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.is_closed = False

    def add_frame(self, fig=None, data=None, cmap=None, dpi=None):
        '''Add a frame to the animation.

        Parameters
        ----------
        fig : Matplotlib figure
            The Matplotlib figure acting as the animation frame.
        data : ndarray
            The image data array acting as the animation frame.
        cmap : Matplotlib colormap
            The optional colormap for the image data.
        dpi : integer or None
            The number of dots per inch with which to save the matplotlib figure.
            If it is not given, the default Matplotlib dpi will be used.

        Raises
        ------
        RuntimeError
            If the function was called on a closed FrameWriter.
        '''
        if self.is_closed:
            raise RuntimeError('Attempted to add a frame to a closed GifWriter.')

        dest = os.path.join(self.path, self.filename.format(self.num_frames))

        if data is None:
            if fig is None:
                fig = plt.gcf()

            facecolor = list(fig.get_facecolor())
            facecolor[3] = 1

            fig.savefig(dest, transparent=False, dpi=dpi, facecolor=facecolor)
        else:
            img = _data_to_img(data, cmap)

            imageio.imwrite(dest, img)

        self.num_frames += 1

    def close(self):
        '''Closes the animation writer.

        Makes sure that no more frames can be written to the directory.
        '''
        self.is_closed = True

class GifWriter(object):
    '''A writer of gif files from Matplotlib figures.

    .. warning::
        This class used to write out individual frames to a directory,
        before converting this into a gif file. This is now done internally.

    Parameters
    ----------
    filename : string
        The path and filename of the gif.
    framerate : integer
        The number of frames per second of the generated gif file.
    '''
    def __init__(self, filename, framerate=15):
        self.is_closed = False
        self.filename = filename
        self.framerate = framerate

        self._frames = []

    @property
    def num_frames(self):
        return len(self._frames)

    def __del__(self):
        try:
            self.close()
        except Exception:
            import warnings
            warnings.warn('Something went wrong while closing the GifWriter...', RuntimeWarning)

    def add_frame(self, fig=None, data=None, cmap=None, dpi=None):
        '''Add a frame to the animation.

        Parameters
        ----------
        fig : Matplotlib figure
            The Matplotlib figure acting as the animation frame.
        data : ndarray
            The image data array acting as the animation frame.
        cmap : Matplotlib colormap
            The optional colormap for the image data.
        dpi : integer or None
            The number of dots per inch with which to save the matplotlib figure.
            If it is not given, the default Matplotlib dpi will be used.

        Raises
        ------
        RuntimeError
            If the function was called on a closed GifWriter.
        '''
        if self.is_closed:
            raise RuntimeError('Attempted to add a frame to a closed GifWriter.')

        if data is None:
            if fig is None:
                fig = plt.gcf()

            facecolor = list(fig.get_facecolor())
            facecolor[3] = 1

            buf = io.BytesIO()

            fig.savefig(buf, transparent=False, dpi=dpi, facecolor=facecolor)

            frame = imageio.imread(buf.getvalue())
        else:
            frame = _data_to_img(data, cmap, copy=True)

        self._frames.append(frame)

    def convert(self):
        '''Convert all frames into a gif file.
        '''
        imageio.mimsave(self.filename, self._frames, duration=1 / self.framerate)

    def close(self):
        '''Close the animation and create the final gif file.
        '''
        try:
            if not self.is_closed:
                self.convert()
        finally:
            self.is_closed = True

class FFMpegWriter(object):
    '''A writer of video files from Matplotlib figures.

    This class uses FFMpeg as the basis for writing frames to a video file.

    Parameters
    ----------
    filename : string
        The filename of the generated video file.
    codec : string
        The codec for FFMpeg to use. If it is not given, a suitable codec
        will be guessed based on the file extension.
    framerate : integer
        The number of frames per second of the generated video file.
    quality : string
        The quality of the encoding for lossy codecs. Please refer to FFMpeg documentation.
    preset : string
        The preset for the quality of the encoding. Please refer to FFMpeg documentation.

    Raises
    ------
    ValueError
        If the codec was not given and could not be guessed based on the file extension.
    RuntimeError
        If something went wrong during initialization of the call to FFMpeg. Most likely,
        FFMpeg is not installed and/or not available from the commandline.
    '''
    def __init__(self, filename, codec=None, framerate=24, quality=None, preset=None):
        if codec is None:
            extension = os.path.splitext(filename)[1]
            if extension == '.mp4' or extension == '.avi':
                codec = 'libx264'
            elif extension == '.webm':
                codec = 'libvpx-vp9'
            else:
                raise ValueError('No codec was given and it could not be guessed based on the file extension.')

        self.is_closed = True
        self.filename = filename
        self.codec = codec
        self.framerate = framerate

        ffmpeg_path = Configuration().plotting.ffmpeg_path
        if ffmpeg_path is None:
            ffmpeg_path = 'ffmpeg'

        if shutil.which(ffmpeg_path) is None:
            raise RuntimeError('ffmpeg was not found. Did you install it and is it accessible, either from PATH or from the HCIPy configuration file?')

        if codec == 'libx264':
            if quality is None:
                quality = 10

            if preset is None:
                preset = 'veryslow'

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', preset, '-r',
                str(framerate), '-crf', str(quality), filename
            ]
        elif codec == 'mpeg4':
            if quality is None:
                quality = 4

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-',
                '-vcodec', 'mpeg4', '-q:v', str(quality), '-r', str(framerate), filename
            ]
        elif codec == 'libvpx-vp9':
            if quality is None:
                quality = 30

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-'
            ]

            if quality < 0:
                command.extend(['-vcodec', 'libvpx-vp9', '-lossless', '1', '-r', str(framerate), filename])
            else:
                command.extend(['-vcodec', 'libvpx-vp9', '-crf', str(quality), '-b:v', '0', '-r', str(framerate), filename])
        else:
            raise ValueError('Codec unknown.')

        try:
            self.p = Popen(command, stdin=PIPE)
        except OSError:
            raise RuntimeError('Something went wrong when opening FFMpeg.')

        self.is_closed = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            import warnings
            warnings.warn('Something went wrong while closing FFMpeg...', RuntimeWarning)

    def add_frame(self, fig=None, data=None, cmap=None, dpi=None):
        '''Add a frame to the animation.

        Parameters
        ----------
        fig : Matplotlib figure
            The Matplotlib figure acting as the animation frame.
        data : ndarray
            The image data array acting as the animation frame.
        cmap : Matplotlib colormap
            The optional colormap for the image data.
        dpi : integer or None
            The number of dots per inch with which to save the matplotlib figure.
            If it is not given, the default Matplotlib dpi will be used.

        Raises
        ------
        RuntimeError
            If the function was called on a closed FFMpegWriter.
        '''
        if self.is_closed:
            raise RuntimeError('Attempted to add a frame to a closed FFMpegWriter.')

        if data is None:
            if fig is None:
                fig = plt.gcf()

            facecolor = list(fig.get_facecolor())
            facecolor[3] = 1

            fig.savefig(self.p.stdin, format='png', transparent=False, dpi=dpi, facecolor=facecolor)
        else:
            img = _data_to_img(data, cmap)

            imageio.imwrite(self.p.stdin, img, format='png')

    def close(self):
        '''Close the animation writer and finish the video file.

        This closes the FFMpeg call.
        '''
        if not self.is_closed:
            self.p.stdin.close()
            self.p.wait()
            self.p = None
        self.is_closed = True

    def _repr_html_(self):
        '''Get an HTML representation of the generated video.

        Helper function for Jupyter notebooks. The video will be inline embedded in an
        HTML5 video tag using base64 encoding. This is not very efficient, so only use this
        for small video files.

        The FFMpegWriter must be closed for this function to work.

        Raises
        ------
        RuntimeError
            If the call was made on an open FFMpegWriter.
        '''
        if not self.is_closed:
            raise RuntimeError('Attempted to show the generated movie on an opened FFMpegWriter.')

        video = open(self.filename, 'rb').read()
        video = base64.b64encode(video).decode('ascii').rstrip()

        if self.filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif self.filename.endswith('.webm'):
            mimetype = 'video/webm'
        elif self.filename.endswith('.avi'):
            mimetype = 'video/avi'
        else:
            raise RuntimeError('Mimetype could not be guessed.')

        output = '''<video controls><source src="data:{0};base64,{1}" type="{0}">Your browser does not support the video tag.</video>'''
        output = output.format(mimetype, video)

        return output
