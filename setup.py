from setuptools import setup, find_packages

# read the contents of the README.md file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()

setup(name='hcipy',
	use_scm_version=True,
	description='A framework for performing optical propagation simulations, meant for high contrast imaging, in Python.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/ehpor/hcipy/',
	author='Emiel Por',
	author_email='por@strw.leidenuniv.nl',
	packages=find_packages(),
	setup_requires=[
		'setuptools_scm'],
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib>=2.0.0",
		"Pillow",
		"pyyaml",
		"mpmath",
		"astropy",
		"imageio",
		"xxhash"],
	zip_safe=False,
	classifiers=(
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3.5",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Topic :: Scientific/Engineering :: Astronomy"
	),
	license='MIT',
	license_file='LICENSE'
)