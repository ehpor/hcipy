import numpy as np

def parse_schott_glass_catalogue(path):
	data = np.loadtxt(path, dtype=np.string, delimiter=',')

	database = {}
	for di in data:
		database[di[0]] = make_sellmeier_glass([1, di[1:4].astype(np.float), di[4::].astype(np.float)])

	return database

def make_sellmeier_glass(A, K, L):
	'''The Sellmeier equation for the dispersion of the refractive index of materials.

	Parameters
	----------
	A : scalar
		The Sellmeier coefficient

	K : array_like
		The Sellmeier coefficients

	L : array_like
		The Sellmeier coefficients

	Returns
	-------
	function 
		The refractive index profile function
	'''
	def refractive_index(wavelength):
		n_squared = A
		for Ki, Li in zip(K,L):
			n_squared += Ki * wavelength**2 / (wavelength**2 - Li)
		return np.sqrt(n_squared)

	return refractive_index

def make_cauchy_glass(coefficients):
	'''The Cauchy equation for the dispersion of the refractive index of materials.

	Parameters
	----------
	A : scalar
		The Sellmeier coefficient
		
	K : array_like
		The Sellmeier coefficients

	L : array_like
		The Sellmeier coefficients

	Returns
	-------
	function 
		The refractive index profile function
	'''
	def refractive_index(wavelength):
		n = 0
		for i, coef in enumerate(coefficients):
			n += coef / wavelength**(2*i)
		return n

	return refractive_index

glass_catalogue = {
	'IP_DIP' : make_cauchy_glass([1.5273, 6.5456E-3, 2.5345E-4]),
}

glass_catalogue['SCHOTT'] = parse_schott_glass_catalogue('../../data/schott_glass_catalogue_2018_09.csv')