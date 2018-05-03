import os
import warnings
import yaml

class Configuration(object):
	def __init__(self):
		if not hasattr(Configuration, '_config'):
			self.reset()
	
	def __getitem__(self, key):
		return Configuration._config[key]
	
	def __setitem__(self, key, value):
		Configuration._config[key] = value
	
	def reset(self):
		paths = ['', '~/.hcipy/']
		paths = [os.path.expanduser(path + 'config.yaml') for path in paths]

		p = None
		for path in paths:
			if os.path.exists(path):
				p = path
				break
		
		if p is None:
			warnings.warn("Configuration file was not found. A new one will be created.")
			
			directory = os.path.dirname(paths[-1])
			if not os.path.exists(directory):
				os.makedirs(directory)

			f = open(paths[-1], 'w')
			f.write(default_configuration)
			f.close()

			p = paths[-1]
		
		with open(p) as f:
			Configuration._config = yaml.load(f.read())

default_configuration = '''# This is the configuration file for the high-contrast imaging for Python (HCIPy) library.

'''