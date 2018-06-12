import numpy as np
from ..propagation import FraunhoferPropagator
from ..field import Field

def generate_app_keller(wavefront, propagator, propagator_max, contrast, num_iterations):
	"""
	Modified Gerchberg-Saxton algorithm based on code by Christoph Keller.
	"""
	wf = wavefront.copy()

	for i in range(num_iterations):
		image = propagator.forward(wf)
		image_max = propagator_max.forward(wf)

		mask = image.intensity / image_max.intensity.max() > contrast

		if not np.any(mask):
			break
		
		delta_image = image.copy()
		delta_image.electric_field[~mask] = 0
		delta_image.electric_field[mask] *= -1#(1 - image_max.amplitude.max() * contrast[mask] / image.amplitude[mask])

		delta_aperture = propagator.backward(delta_image)
		wf.electric_field += delta_aperture.electric_field

		epsilon = 1e-6
		m = wavefront.amplitude < epsilon
		wf.electric_field[m] = 0
		wf.electric_field[~m] *= wavefront.amplitude[~m] / wf.amplitude[~m]
	
	return wf

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

	M = propagator.get_transformation_matrix_forward(wavefront.wavelength)
	M_max = propagator_max.get_transformation_matrix_forward(wavefront.wavelength)

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