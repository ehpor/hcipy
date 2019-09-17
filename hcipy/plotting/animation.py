import glob
import os
from subprocess import Popen, PIPE

import matplotlib
import imageio
from PIL import Image


class GifWriter(object):
	def __init__(self, filename, framerate=15):
		self.closed = False
		self.filename = filename
		self.framerate = framerate

		self.path_to_frames = self.filename + "_frames"
		if not os.path.exists(self.path_to_frames):
			os.mkdir(self.path_to_frames)
		self.n_frames = 0
	
	def add_frame(self, fig=None, arr=None, cmap=None):
		if self.closed:
			raise RuntimeError('Attempted to add a frame to a closed GifWriter.')

		dest = os.path.join(self.path_to_frames, '%05d.png' % self.n_frames)

		if arr is None:
			if fig is None:
				fig = matplotlib.pyplot.plt.gcf()
			fig.savefig(dest, format='png', transparent=False)
		else:
			if not cmap is None:
				arr = matplotlib.cm.get_cmap(cmap)(arr, bytes=True)
			
			imageio.imwrite(dest, arr, format='png')

		self.n_frames += 1

	def convert2gif(self):
		search_pattern = os.path.join(self.path_to_frames, "*.png")
		files = glob.glob(search_pattern)
		files.sort()

		if len(files) != self.n_frames:
			raise OSError("Expected {} files but found {}".format(self.n_frames, len(files)))

		# Open all frames to convert
		frames = []
		for file in files:
			frames.append(Image.open(file))

		# Convert to GIF
		# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html?highlight=duration#saving
		# duration := display duration of each frame in ms
		duration = int(1000 / self.framerate)
		frames[0].save(self.filename,
						format="GIF",
						append_images=frames[1:],
						save_all=True,
						duration=duration,
						loop=0)

	def close(self):
		try:
			if not self.closed:
				self.convert2gif()
		finally:
			self.n_frames = 0
			self.closed = True


class FFMpegWriter(object):
	def __init__(self, filename, codec=None, framerate=24, quality=None, preset=None):
		if codec is None:
			extension = os.path.splitext(filename)[1]
			if extension == '.mp4':
				codec = 'mpeg4'
			elif extension == '.avi':
				codec = 'libx264'
			else:
				raise ValueError('No codec was given and it could not be guessed based on file extension.')

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
		else:
			raise ValueError('Codec unknown.')
		
		self.p = Popen(command, stdin=PIPE)
		self.closed = False

	def add_frame(self, fig=None, arr=None, cmap=None):
		if self.closed:
			raise RuntimeError('Attempted to add a frame to a closed FFMpegWriter.')

		if arr is None:
			if fig is None:
				fig = plt.gcf()
			fig.savefig(self.p.stdin, format='png', transparent=False)
		else:
			if not cmap is None:
				arr = matplotlib.cm.get_cmap(cmap)(arr, bytes=True)
			
			imageio.imwrite(self.p.stdin, arr, format='png')

	def close(self):
		if not self.closed:
			self.p.stdin.close()
			self.p.wait()
			self.p = None
		self.closed = True