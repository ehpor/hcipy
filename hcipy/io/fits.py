import numpy as np 

from ..field import Field, make_pupil_grid 
from ..mode_basis import ModeBasis

def read_fits(filename):
	from astropy.io import fits
	return fits.getdata(filename)
	
def write_fits(data, filename, shape=None):
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
    will be used for the grid generation (if the modes are square arrays and assuming that the 
    most used mode basis is defined in the pupil plane).  
    
    Parameters
    ----------
    filename : string
        The name of the fits file.
    grid : Grid
        If given, the grid on which the modes are sampled.  
    
    '''

    # loading the data cube which contains the mode basis. 
    data_cube = read_fits(filename)

    # list where the modes will be stored
    modes = []

    # setting the grid 
    if grid != None: 
        input_grid = grid
    else:
        if data_cube.shape[1] == data_cube.shape[2]:
            # if modes are square we generate a pupil grid 
            input_grid = make_pupil_grid(data_cube.shape[1])
        else:
            # otherwise an error is raised 
            raise ValueError('Modes are non-square arrays, user must pass a grid as argument.')

    # loading the modes as Fields and adding them to the list 
    for i in np.arange(data_cube.shape[0]):
        modes.append(Field(data_cube[i,:,:].ravel(), input_grid))

    # creating the proper mode basis 
    mode_basis = ModeBasis(modes)

    return mode_basis

