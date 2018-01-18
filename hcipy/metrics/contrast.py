import numpy as np

def get_strehl_from_focal(img, ref_img):
	'''Get the Strehl ratio from a focal-plane image.

	Parameters
	----------
	img : Field or array_like
		The focal-plane image.
	ref_img : Field or array_like
		The reference focal-plane image without aberrations.

	Returns
	-------
	scalar
		The Strehl ratio.
	'''
	return img(np.argmax(ref_img)) / ref_img.max()

def get_strehl_from_pupil(aperture, ref_aperture):
	'''Get the Strehl ratio from a pupil-plane electric field.

	Parameters
	----------
	aperture : Field or array_like
		The pupil-plane electric field.
	ref_aperture : Field or array_like
		The reference pupil-plane electric field without aberrations.

	Returns
	-------
	scalar
		The Strehl ratio.
	'''
	return np.abs(np.sum(aperture) / np.sum(ref_aperture))**2

def get_mean_intensity_in_roi(img, mask):
	'''Get the mean intensity in a masked region of interest.

	Parameters
	----------
	img : Field or array_like
		The focal-plane image.
	mask : Field or array_like
		A binary array describing the region of interest.
	
	Returns
	-------
	scalar
		The mean intensity in the region of interest.
	'''
	return np.mean(img[mask])

def get_mean_raw_contrast(img, mask, ref_img):
	'''Get the mean raw contrast in a masked region of interest.

	Parameters
	----------
	img : Field or array_like
		The focal-plane image.
	mask : Field or array_like
		A binary array describing the region of interest.
	img_ref : Field or array_like
		A reference focal-plane image without aberrations. This is used
		to determine the Strehl ratio.
	
	Returns
	-------
	scalar
		The mean raw contrast in the region of interest.
	'''
	mean_intensity = get_mean_intensity_in_roi(img, mask)
	strehl = get_strehl_from_focal(img, ref_img)

	return mean_intensity / strehl