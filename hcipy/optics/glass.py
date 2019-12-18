import numpy as np

def make_sellmeier_glass(A, K, L):

	def refractive_index(wavelength):
		n_squared = A
		for Ki, Li in zip(K,L):
			n_squared += Ki * wavelength**2 / (wavelength**2 - Li)
		return np.sqrt(n_squared)

	return refractive_index

def make_cauchy_glass(coefficients):

	def refractive_index(wavelength):
		n = 0
		for i, coef in enumerate(coefficients):
			n += coef / wavelength**(2*i)
		return n

	return refractive_index

glass_catalogue = {
	'IP_DIP' : make_cauchy_glass([1.5273, 6.5456E-3, 2.5345E-4]),
}