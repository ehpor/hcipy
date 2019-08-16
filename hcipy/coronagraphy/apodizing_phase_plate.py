import numpy as np
from ..propagation import FraunhoferPropagator
from ..field import Field

def generate_app_keller(wavefront, propagator, contrast, num_iterations, beta=0):
	"""
	Accelerated Gerchberg-Saxton-like algorithm for APP design by
	Christoph Keller [1]_ and based on Douglas-Rachford operator splitting.
	The acceleration was inspired by the paper by Jim Fienup [2_]. The 
	acceleration can provide speed-ups of up to two orders of magnitude and 
	produce better APPs.

	.. [1] Keller C.U., 2016, "Novel instrument concepts for
		characterizing directly imaged exoplanets", Proc. SPIE 9908,
		Ground-based and Airborne Instrumentation for Astronomy VI, 99089V
		doi: 10.1117/12.2232633; https://doi.org/10.1117/12.2232633
	.. [2] J. R. Fienup, 1976, "Reconstruction of an object from the modulus 
		of its Fourier transform," Opt. Lett. 3, 27-29

	Parameters
	----------
	wavefont : Wavefront
		The input aperture as a wavefront; a phase can be provided as a
		starting point for the APP generation.
	propagator : Propagator
		The propagator from the pupil grid to the focal plane grid.
	contrast : array_like
		The required contrast in the focal plane; this is a float mask
		that is 1.0 everywhere except for the dark zone where it is the
		contrast such as 1e-5.
	num_iterations : int
		The maximum number of iterations.
	beta : scalar
		The acceleration parameter. The default is 0 (no acceleration).
		Good values for beta are typically between 0.3 and 0.9. Values larger
		than 1.0 will not work.

	Returns
	-------
	Wavefront
		The APP as a wavefront.
	
	Raises
	------
	ValueError
		If beta is not between 0 and 1.
	"""
	if beta < 0 or beta > 1:
		raise ValueError('Beta should be between 0 and 1.')

	# initialize APP with wavefront
	app = wavefront.copy()

	# define pupil aperture where amplitude is larger than 1e-6
	aperture = wavefront.amplitude > 1e-6

	# define dark zone as location where contrast requirement is < 1e-1
	dark_zone = contrast < 0.1

	for i in range(num_iterations):
		# calculate image plane electric field
		image = propagator.forward(app)

		# stop iteration if contrast requirement is met
		if not np.any(image.intensity / image.intensity.max() > contrast):
			break

		# modify focal plane electic field using acceleration
		new_image = image.copy()
		if beta != 0 and i > 1:
			new_image.electric_field[dark_zone] = (
				old_image.electric_field[dark_zone] * beta
				- new_image.electric_field[dark_zone] * (1 + beta))
		else:
			new_image.electric_field[dark_zone] = 0
		old_image = new_image.copy()

		# determine corresponding aperture electric field
		app.electric_field = propagator.backward(new_image).electric_field

		# enforce aperture support
		app.electric_field[~aperture] = 0

		# enforce unity transmission within aperture support
		app.electric_field[aperture] *= (
			app.amplitude[aperture] / app.amplitude[aperture])

	return app

def generate_app_por(wavefront, propagator, propagator_max, contrast, num_iterations=1):
	'''Optimize a one-sided APP using a globally optimal algorithm. This algorithm does not
	apply any symmetries for two-sided dark zones or circularly-symmetric pupils. This
	function requires that you have installed the Gurobi optimizer.

	Parameters
	----------
	wavefont : Wavefront
		The input aperture as a wavefront.
	propagator : Propagator
		The propagator from `wavefront` to the focal plane grid.
	propagator_max : Propagator
		The propagator from `wavefront` to the point in the focal-plane that we want to maximize.
	contrast : array_like or scalar
		The required contrast in the focal plane.
	num_iterations : int
		The number of iterations for the Strehl ratio to converge.

	Returns
	-------
	Field
		The resultant electric field transmission function.
	'''
	import gurobipy as gp

	M = propagator.get_transformation_matrix_forward(wavefront.electric_field.grid, wavefront.wavelength)
	M_max = propagator_max.get_transformation_matrix_forward(wavefront.electric_field.grid, wavefront.wavelength)

	wf_0 = propagator.forward(wavefront).electric_field
	M /= wf_0[:,np.newaxis]

	M *= wavefront.electric_field
	M_max *= wavefront.electric_field

	M_max = np.sum(M_max, axis=0)

	M = np.array(np.bmat([[M.real, -M.imag], [M.imag, M.real]]))
	M_max = np.concatenate((M_max.real, -M_max.imag))

	m, n = M.shape
	contrast_requirement = np.ones(m) * np.sqrt(contrast)

	model = gp.Model('lp')
	model.Params.Threads = 4

	x = model.addVars(n, lb=-1, ub=1)

	obj = gp.quicksum((x[i] * M_max[i] for i in range(n)))
	model.setObjective(obj, gp.GRB.MAXIMIZE)

	for i in range(n//2):
		r2 = x[i]*x[i] + x[i+n//2]*x[i+n//2]
		model.addQConstr(r2 <= 1)

	for i, ee in enumerate(M):
		e = gp.quicksum((x[i] * ee[i] for i in range(n)))
		model.addConstr(e <= contrast_requirement[i])
		model.addConstr(e >= -contrast_requirement[i])

	model.optimize()

	solution = np.array([x[i].x for i in range(n)])
	solution = solution[:n//2] + 1j * solution[n//2:]

	return Field(solution, wavefront.electric_field.grid)

def generate_app_doelman(wavefront, propagator, propagator_max, contrast, num_iterations):
	raise NotImplementedError()
