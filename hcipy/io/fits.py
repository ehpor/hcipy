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

