import glob
import os
import shutil
import base64
from subprocess import Popen, PIPE

import matplotlib
import imageio
from PIL import Image

class GifWriter(object):
	'''A writer of gif files from Matplotlib figures.

	This function writes out individual frames to a `filename.frames` directory. When
	the `close()` function is called, or the object is removed by the garbage collector,
	the frames are collected in a single gif file. The individual frames are then deleted,
	if `cleanup` is True (default).

	Parameters
	----------
	framerate : integer
		The number of frames per second of the generated gif file.
	cleanup : boolean
		Whether to clean up the generated frames.
	'''
	def __init__(self, filename, framerate=15, cleanup=True):
		self.closed = False
		self.filename = filename
		self.framerate = framerate
		self.cleanup = cleanup

		self.path_to_frames = self.filename + "_frames"
		if not os.path.exists(self.path_to_frames):
			os.mkdir(self.path_to_frames)
		self.num_frames = 0

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass

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
		if self.closed:
			raise RuntimeError('Attempted to add a frame to a closed GifWriter.')

		dest = os.path.join(self.path_to_frames, '%05d.png' % self.num_frames)

		if data is None:
			if fig is None:
				fig = matplotlib.pyplot.gcf()
			fig.savefig(dest, format='png', transparent=False)
		else:
			if cmap is not None:
				data = matplotlib.cm.get_cmap(cmap)(data, bytes=True)
			
			imageio.imwrite(dest, data, format='png')

		self.num_frames += 1

	@staticmethod
	def convert_to_gif(dest_filename, src_file_path, framerate, src_file_suffix="png", num_files_to_convert=None):
		'''Helper function to convert all files in a directory to a gif file.

		Parameters
		----------
		dest_filename : string
			The filename for the gif file.
		src_file_path : string
			The path to the directory with all the frames as image files.
		framerate : integer
			The number of frames per second for the generated gif file.
		src_file_suffix : string
			The file extension of the image files.
		num_files_to_convert : integer or None
			How many frames are expected in the directory. If None, then no check will be done.
		
		Raises
		------
		OSError
			If the number of files in the directory differs from the number of expected files.
		'''
		search_pattern = os.path.join(src_file_path, "*."+src_file_suffix)
		files = glob.glob(search_pattern)
		files.sort()

		if num_files_to_convert is not None and len(files) != num_files_to_convert:
			raise OSError("Expected {} files but found {}".format(num_files_to_convert, len(files)))

		# Open all frames to convert
		frames = []
		for image_file in files:
			frames.append(Image.open(image_file))

		# Convert to GIF
		# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html?highlight=duration#saving
		# duration := display duration of each frame in ms
		duration = int(1000 / framerate)
		frames_to_append = frames[1:] if len(frames) > 1 else []
		frames[0].save(dest_filename,
						format="GIF",
						append_images=frames_to_append,
						save_all=True,
						duration=duration,
						loop=0)

	def convert(self):
		'''Convert all images in the frames directory into a gif file.

		This function doesn't remove the frames after conversion.
		'''
		return self.convert_to_gif(self.filename, self.path_to_frames, self.framerate, num_files_to_convert=self.num_frames)

	def close(self):
		'''Close the animation, create the final gif file and (potentially) remove the individual frames.
		'''
		try:
			if not self.closed:
				self.convert()
				if self.cleanup:
					shutil.rmtree(self.path_to_frames, ignore_errors=True)
		finally:
			self.num_frames = 0
			self.closed = True

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

		self.closed = True
		self.filename = filename
		self.codec = codec
		self.framerate = framerate
		
		if codec == 'libx264':
			if quality is None:
				quality = 10
			if preset is None:
				preset = 'veryslow'
			command = ['ffmpeg', '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe', 
				'-vcodec','png', '-r', str(framerate), '-i', '-']
			command.extend(['-vcodec', 'libx264', '-preset', preset, '-r', 
				str(framerate), '-crf', str(quality), filename])
		elif codec == 'mpeg4':
			if quality is None:
				quality = 4
			command = ['ffmpeg', '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe', 
				'-vcodec','png', '-r', str(framerate), '-i', '-']
			command.extend(['-vcodec', 'mpeg4', '-q:v', str(quality), '-r', 
				str(framerate), filename])
		elif codec == 'libvpx-vp9':
			if quality is None:
				quality = 30
			command = ['ffmpeg', '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe', 
				'-vcodec','png', '-r', str(framerate), '-i', '-']
			if quality < 0:
				command.extend(['-vcodec', 'libvpx-vp9', '-lossless', '1',
					'-r', str(framerate), filename])
			else:
				command.extend(['-vcodec', 'libvpx-vp9', '-crf', str(quality),
					'-b:v', '0', '-r', str(framerate), filename])
		else:
			raise ValueError('Codec unknown.')

		try:
			self.p = Popen(command, stdin=PIPE)
		except OSError as err:
			raise RuntimeError('Something went wrong when opening FFMpeg. Is FFMpeg installed and accessible from the command line?')
		self.closed = False

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass

	def add_frame(self, fig=None, arr=None, cmap=None, dpi=None):
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
		if self.closed:
			raise RuntimeError('Attempted to add a frame to a closed FFMpegWriter.')

		if arr is None:
			if fig is None:
				fig = matplotlib.pyplot.gcf()
			fig.savefig(self.p.stdin, format='png', transparent=False, dpi=dpi)
		else:
			if not cmap is None:
				arr = matplotlib.cm.get_cmap(cmap)(arr, bytes=True)
			
			imageio.imwrite(self.p.stdin, arr, format='png')

	def close(self):
		'''Close the animation writer and finish the video file.

		This closes the FFMpeg call.
		'''
		if not self.closed:
			self.p.stdin.close()
			self.p.wait()
			self.p = None
		self.closed = True

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
		if not self.closed:
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
