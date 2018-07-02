from setuptools import setup, find_packages

setup(name='hcipy',
	version='0.1',
	description='A framework for performing optical propagation simulations, meant for high contrast imaging, in Python.',
	url='https://gitlab.strw.leidenuniv.nl/por/hcipy',
	author='Emiel Por',
	author_email='por@strw.leidenuniv.nl',
	packages=find_packages(),
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib>=2.0.0",
		"Pillow",
		"progressbar2",
		"pyyaml"],
	zip_safe=False,
	classifiers=(
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Astronomy"
	)
)