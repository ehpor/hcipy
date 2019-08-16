import matplotlib.pyplot as plt
import matplotlib
from subprocess import Popen, PIPE, call
import imageio
import os

class GifWriter(object):
	def __init__(self, filename, framerate=15):
		self.closed = False
		self.filename = filename
		self.framerate = framerate

		if not os.path.exists(filename + '_frames/'):
			os.mkdir(filename + '_frames/')
		self.frame_number = 0
	
	def add_frame(self, fig=None, arr=None, cmap=None):
		if self.closed:
			raise RuntimeError('Attempted to add a frame to a closed GifWriter.')
		
		dest = self.filename + '_frames/%05d.png' % self.frame_number
		self.frame_number += 1

		if arr is None:
			if fig is None:
				fig = plt.gcf()
			fig.savefig(dest, format='png', transparent=False)
		else:
			if not cmap is None:
				arr = matplotlib.cm.get_cmap(cmap)(arr, bytes=True)
			
			imageio.imwrite(dest, arr, format='png')
	
	def close(self):
		if not self.closed:
			command = ['convert', '-delay', str(int(100/self.framerate)), 
						'-loop', '0', self.filename + '_frames/*.png', self.filename]
			call(command)	
			call(['rm', '-rf', self.filename + '_frames'])
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