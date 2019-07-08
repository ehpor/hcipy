import numpy as np 

from ..field import Field, make_pupil_grid 
from ..mode_basis import ModeBasis

def read_fits(filename):
	'''Read an array from a fits file.

	Parameters
	----------
	filename : string
		The filename of the file to read. This can include a path.

	Returns
	-------
	ndarray
		The ndarray read from the fits file.
	'''
	from astropy.io import fits
	return fits.getdata(filename).copy()
	
def write_fits(data, filename, shape=None, overwrite=True):
	'''Write the data to a fits-file.

	If the data is a Field with SeparatedCoords, the shaped field will be written to the
	fits-file, to allow for easier viewing with external tools.

	Parameters
	----------
	data : ndarray or Field
		The ndarray or Field to write to the fits file.
	filename : string
		The filename of the newly created file. This can include a path.
	shape : ndarray or None
		The shape to which to reshape the data. If this is given, it will override a potential
		shape from the grid accompaning the field.
	overwrite : boolean
		Whether to overwrite the fits-file if it already exists.
	'''
	from astropy.io import fits
	
	hdu = None
	
	if shape is not None:
		hdu = fits.PrimaryHDU(data.reshape(shape))
	elif hasattr(data, 'grid'):
		if data.grid.is_separated:
			hdu = fits.PrimaryHDU(data.shaped)
	
	if hdu is None:
		hdu = fits.PrimaryHDU(data)

	hdu.writeto(filename, overwrite=True)
"""
def write_mode_basis(mode_basis, filename):
	'''A function that writes a mode basis as fits file.  

	Parameters
	----------
	mode_basis : ModeBasis
		The mode basis that will be saved as fits file.
	filename : string
		The name that the fits file will get.
	'''
	# the number of modes in the basis 
	Nmodes = len(mode_basis)

	# the shape of the field if it were shaped
	shape = mode_basis[0].grid.shape

	write_fits(np.array([mode_basis]), filename, shape=[Nmodes, shape[0], shape[1]])

def read_mode_basis(filename, grid=None):
	'''A function that reads a saved mode basis fits file as a proper mode basis. 

	Reads a mode basis fits file, assuming that it has been saved in the format by write_mode_basis().
	A grid on which the modes are sampled can be given, otherwise the function make_pupil_grid() 
	will be used for the grid generation. 
	
	Parameters
	----------
	filename : string
		The name of the fits file.
	grid : Grid
		If given, the grid on which the modes are sampled.

	Returns
	-------
	ModeBasis
		The read-in ModeBasis from the file.
	'''
	# Load the data cube which contains the mode basis. 
	data_cube = read_fits(filename)

	# Sett a default grid if none is provided
	if grid is None:
		grid = make_pupil_grid(data_cube.shape[2:0:-1])

	# Load the modes as Fields and adding them to the list
	modes = []
	for i in np.arange(data_cube.shape[0]):
		modes.append(Field(data_cube[i,:,:].ravel(), grid))

	# Return the proper mode basis 
	return ModeBasis(modes)
"""