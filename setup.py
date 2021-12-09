from setuptools import setup, find_packages

# read the contents of the README.md file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()

setup(
	name='hcipy',
	use_scm_version=True,
	description='A framework for performing optical propagation simulations, meant for high contrast imaging, in Python.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/ehpor/hcipy/',
	author='Emiel Por',
	author_email='epor@stsci.edu',
	packages=find_packages(),
	package_data={'hcipy': ['*.yaml', '*.csv', '.fits']},
	setup_requires=[
		'setuptools_scm'
	],
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib>=2.0.0",
		"Pillow",
		"pyyaml",
		"astropy",
		"imageio",
		"xxhash",
		"numexpr",
		"asdf",
		"importlib_metadata ; python_version<'3.7'",
		"importlib_resources>=1.4 ; python_version<'3.9'"
	],
	extras_require={
		"dev": [
			"pytest",
			"codecov",
			"coverage",
			"mpmath",
			"dill",
			"flake8"],
		"doc": [
			"numpydoc",
			"sphinx_rtd_theme",
			"nbsphinx",
			"jupyter_client",
			"ipykernel",
			"poppy",
			"nbclient",
			"nbformat",
			"nbconvert",
			"sphinx-automodapi",
			"progressbar2"]},
	zip_safe=False,
	classifiers=[
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Topic :: Scientific/Engineering :: Astronomy"],
	license_files=["LICENSE"]
)
