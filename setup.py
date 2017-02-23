from setuptools import setup, find_packages

setup(name='hcipy',
	version='0.1',
	description='A framework for performing optical propagation simulations, meant for high contrast imaging, in Python.',
	url='https://gitlab.strw.leidenuniv.nl/por/hcipy',
	author='Emiel Por',
	author_email='por@strw.leidenuniv.nl',
	packages=find_packages(),
	zip_safe=False)
