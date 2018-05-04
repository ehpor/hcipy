import os
import warnings
import yaml

class Configuration(object):
	'''A configuration object that describes the current configuration status of the package.
	'''
	def __init__(self):
		if not hasattr(Configuration, '_config'):
			self.reset()
	
	def __getitem__(self, key):
		'''Get the value for the key.

		Parameters
		----------
		key : string
			The configuration key.
		'''
		return Configuration._config[key]
	
	def __setitem__(self, key, value):
		'''Set the value for the key.

		Parameters
		----------
		key : string
			The configuration key.
		value : anything
			The value to set this to.
		'''
		Configuration._config[key] = value
	
	def get_config_path(self):
		'''Get the current path of the configuration file.

		If no configuration file is present, a new one will be made, 
		and its path will be returned.

		Returns
		-------
		string
			The configuration directory.
		'''
		paths = ['./']
		if 'XDG_CONFIG_HOME' in os.environ:
			for p in os.environ['XDG_CONFIG_HOME'].split(os.pathsep):
				paths.append(p + '/hcipy/')
		if 'XDG_CONFIG_DIRS' in os.environ:
			for p in os.environ['XDG_CONFIG_DIRS'].split(os.pathsep):
				paths.append(p + '/hcipy/')
		paths.append(os.path.expanduser('~/.config/hcipy/'))
		paths.append(os.path.expanduser('~/.hcipy/'))
		paths = [p + 'hcipy_config.yaml' for p in paths]

		for path in paths:
			if os.path.exists(path):
				return path
		
		# No configuration file is available, make a new one.
		path = paths[-1]
		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		f = open(path, 'w')
		f.write(default_configuration)
		f.close()

		return path	
	
	def reset(self):
		'''Read the configuration file from the configuration directory.
		'''
		with open(self.get_config_path()) as f:
			Configuration._config = yaml.load(f.read())

		if Configuration._config is None:
			Configuration._config = {}

default_configuration = '''# This is the configuration file for the high-contrast imaging for Python (HCIPy) library.

'''