import numpy as np
import inspect
import copy
import collections

class OpticalElement(object):
	'''Base class for all optical elements.

	This class can propagate a :class:`Wavefront` through an optical element
	(or free space), therefore modifying it. Any OpticalElement should be agnostic
	of the grid and wavelength of the wavefront. If it's not, you can use the 
	:func:`make_agnostic_optical_element` decorator to create an agnostic optical
	element out of a gnostic one.
	'''
	def __call__(self, wavefront):
		'''Propagate a wavefront forward through the optical element.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		return self.forward(wavefront)
	
	def forward(self, wavefront):
		'''Propagate a wavefront forward through the optical element.

		This will be implemented by the derived class.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		raise NotImplementedError()
	
	def backward(self, wavefront):
		'''Propagate a wavefront backward through the optical element.

		A backward propagation is a literal backward propagation through
		the element. It is not (in general) the inverse of the forward
		propagation, except in cases where energy is conserved.

		This function will be implemented by the derived class.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		raise NotImplementedError()
	
	def get_transformation_matrix_forward(self, wavelength=1):
		'''Calculate the linear transformation matrix that corresponds 
		to a forward propagation.

		The matrix is defined such that `E_out = M.dot(E_in)`, where `E_out` and 
		`E_in` are the electric fields of the output and input respectively, and
		`M` is the transformation matrix returned by this function.

		:: warning::
			Depending on the chosen resolution, this matrix can be extremely large and
			use extremely large amounts of memory. For example, a Fresnel propagation on
			a 128x128 grid will already create a matrix of 2.1GB. Use with caution.

		This function will be implemented by the derived class.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		raise NotImplementedError()
	
	def get_transformation_matrix_backward(self, wavelength=1):
		'''Calculate the linear transformation matrix that corresponds 
		to a backward propagation.

		The matrix is defined such that `E_in = M.dot(E_out)`, where `E_out` and 
		`E_in` are the electric fields of the output and input plane respectively, and
		`M` is the transformation matrix returned by this function.

		A backward propagation is a literal backward propagation through
		the element. It is not (in general) the inverse of the forward
		propagation, except in cases where energy is conserved.

		:: warning::
			Depending on the chosen resolution, this matrix can be extremely large and
			use extremely large amounts of memory. For example, a Fresnel propagation on
			a 128x128 grid will already create a matrix of 2.1GB. Use with caution.

		This function will be implemented by the derived class.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		raise NotImplementedError()
	
	def get_instance(self, input_grid=None, output_grid=None, wavelength=None):
		'''Return an OpticalElement that can handle wavefronts with input_grid and wavelength.

		While any OpticalElement should in theory be able to handle all grids and wavelengths,
		this function is added to simplify the interface for those who don't. It allows the user
		to always access properties of an OpticalElement, evaluated for a specific input_grid
		and wavelength.

		The user needs to supply at least an input grid or an output grid, and a wavelength. If
		this is not done, a ValueError will be raised.

		Parameters
		----------
		input_grid : Grid
			The grid on which the input wavefront is defined.
		output_grid : Grid or None
			The grid on which the output wavefront is defined.
		wavelength : scalar
			The wavelength on which the wavefront is defined.
		
		Returns
		-------
		OpticalElement
			An optical element that can handle wavefront with the specified input grid and wavelength.
		'''
		return self

class EmptyOpticalElement(OpticalElement):
	'''An empty optical element.
	
	This optical element doesn't modify the wavefront at all. This can be used as a replacement
	for optical elements. For example, when you don't want to use a coronagraph for code that expects
	an optical element as a coronagraph, you can pass an instance of this class to effectively do nothing.
	'''
	def forward(self, wavefront):
		'''Propagate the wavefront forward through the empty optical element.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		return wavefront
	
	def backward(self, wavefront):
		'''Propagate the wavefront backward through the empty optical element.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		return wavefront
	
	def get_transformation_matrix_forward(self, wavelength=1):
		'''Calculate the backward linear transformation matrix for the empty optical element.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		return np.array(1)
	
	def get_transformation_matrix_backward(self, wavelength=1):
		'''Calculate the backward linear transformation matrix for the empty optical element.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		return np.array(1)

def _get_function_parameters(func):
	if hasattr(inspect, 'signature'):
		# Python 3
		return list(inspect.signature(func).parameters.keys())
	else:
		# Python 2
		return inspect.getargspec(func).args

def make_agnostic_optical_element(grid_dependent_arguments=None, wavelength_dependent_arguments=None, num_in_cache=50):
	'''Create an optical element that is agnostic to input_grid or wavelength from one that is not.

	This decorator is meant to simplify the creation of agnostic optical elements. When you have an 
	optical element that explicitly needs an input grid and/or wavelength on initialization, you can
	use this decorator to modify it to make it accept all input grids and/or wavelengths.

	All parameters pass to the __init__() of the returned class will be attempted to be evaluated on
	an input grid or wavelength (depending on if the parameter name appears in `grid_dependent_arguments`
	or `wavelength_dependent_arguments`). This evaluation is done by calling the argument with either
	the input_grid or wavelength, before passing it to the initializer of the optical element provided
	by the user. When an argument can be both dependent on input grid and wavelength, you can pass a
	function with double arguments. This will be evaluated as `arg(input_grid, wavelength)`. If the 
	argument only has a single parameter, this function will make a guess on whether it is input_grid
	or wavelength dependent and try both possibilities.

	Parameters
	----------
	grid_dependent_arguments : list of strings or None
		A list of all names of parameters that could vary with input_grid. These parameters will be 
		evaluated on the grid if they are callable. If this is None, this indicates that no parameters
		can depend on input_grid, and that the optical element was already input_grid agnostic.
	wavelength_dependent_arguments : list of strings or None
		A list of all names of parameters that could vary with wavelength. These parameters will be 
		evaluated at the wavelength if they are callable. If this is None, this indicates that no parameters
		can depend on wavelength, and that the optical element was already wavelength agnostic.
	num_in_cache : int
		The maximum size of the internal cache for optical elements. Reduce this if the cache is using
		too much memory, increase if there are a lot of cache misses.
	
	Returns
	-------
	class
		The new optical element class that is now agnostic to input grid and wavelength.
	
	Raises
	------
	RuntimeError
		If one of the argument gave an error during evaluation.
	'''
	if grid_dependent_arguments is None:
		grid_dependent_arguments = []

	if wavelength_dependent_arguments is None:
		wavelength_dependent_arguments = []
	
	def decorator(optical_element_class):
		gnostic_param_names = _get_function_parameters(optical_element_class.__init__)[1:]

		grid_dependent = grid_dependent_arguments or 'input_grid' in gnostic_param_names
		wavelength_dependent = wavelength_dependent_arguments or 'wavelength' in gnostic_param_names

		class AgnosticOpticalElement(OpticalElement):
			def __init__(self, *args, **kwargs):
				self._cache = collections.OrderedDict()

				self._parameters = dict(zip(gnostic_param_names, args))
				self._parameters.update(kwargs)
			
			def get_instance(self, input_grid=None, output_grid=None, wavelength=None):
				if grid_dependent and ((input_grid is None) == (output_grid is None)):
					raise ValueError('You need to supply either an input or output grid.')
				
				if wavelength_dependent and (wavelength is None):
					raise ValueError('You need to supply a wavelength.')
				
				# Get cache key
				cache_key = ()

				if grid_dependent:
					if input_grid is not None:
						cache_key += ('input', input_grid)
					else:
						cache_key += ('output', output_grid)
				
				if wavelength_dependent:
					# Use approximate wavelength as a key (match if within 1e-9 relatively).
					wavelength_key = int(np.round(np.log(wavelength) / np.log(1 + 1e-9)))
					cache_key += (wavelength_key, )

				# Is there an element in the cache.
				if cache_key in self._cache:
					return self._cache[cache_key]
				
				if output_grid is not None:
					# If we supplied an output grid, it needs to be listed into the cache, as we
					# cannot initialize/create an optical element from an output grid.
					raise RuntimeError('Output grid is not known. Perform a forward propagation first before backwards propagation on the same grid.')
				
				# If the cache is full, remove the oldest element to make room for a new one.
				if len(self._cache) == 2 * num_in_cache:
					self._cache.popitem(False)
					self._cache.popitem(False)

				# Create a new element.
				element_parameters = dict(self._parameters)

				if 'input_grid' in gnostic_param_names:
					element_parameters['input_grid'] = input_grid
				if 'wavelength' in gnostic_param_names:
					element_parameters['wavelength'] = wavelength
				
				# Evaluate grid dependent arguments (including double-dependent arguments)
				for param_name in grid_dependent_arguments:
					if not callable(element_parameters[param_name]):
						# Argument is not callable, so no evaluation can be done.
						continue
					
					if param_name in wavelength_dependent_arguments:
						# Argument can be a function of either or both. Check between either or both first.
						param_parameters = _get_function_parameters(element_parameters[param_name])
						if len(param_parameters) == 1:
							# Argument is a function of either, but we do not know which. Look if we have clues:
							function_of = 'input_grid'
							if 'grid' in param_parameters[0]:
								function_of = 'input_grid'
							elif 'lam' in param_parameters[0] or 'wave' in param_parameters[0] or 'wvl' in param_parameters[0]:
								function_of = 'wavelength'
							
							# Try first choice
							try:
								if function_of == 'input_grid':
									res = element_parameters[param_name](input_grid)
								else:
									res = element_parameters[param_name](wavelength)
							except Exception:
								# Function evaluation failed. Try other one:
								try:
									if function_of == 'input_grid':
										res = element_parameters[param_name](wavelength)
									else:
										res = element_parameters[param_name](input_grid)
								except Exception:
									# Function evaluation failed again. Raise exception.
									raise RuntimeError('The argument %s can not be evaluated.' % param_name)

							element_parameters[param_name] = res
						else:
							# Argument is a function of both.
							try:
								res = element_parameters[param_name](input_grid, wavelength)
							except Exception:
								# Function evaluation failed again. Raise exception.
								raise RuntimeError('The argument %s can not be evaluated.' % param_name)
					else:
						# Argument is a function of input_grid.
						try:
							element_parameters[param_name] = element_parameters[param_name](input_grid)
						except Exception:
							# Function evaluation failed. Raise exception.
							raise RuntimeError('The argument %s can not be evaluated.' % param_name)
				
				# Evaluate wavelength dependent arguments
				for param_name in wavelength_dependent_arguments:
					if not callable(element_parameters[param_name]):
						# Argument is not callable, so no evaluation can be done.
						continue
					
					if param_name in grid_dependent_arguments:
						# Argument already handled above.
						continue

					# Argument is a function of wavelength.
					try:
						element_parameters[param_name] = element_parameters[param_name](wavelength)
					except Exception:
						# Function evaluation failed. Raise exception.
						raise RuntimeError('The argument %s can not be evaluated.' % param_name)
				
				# Create element.
				elem = optical_element_class(**element_parameters)

				# Add element to cache.
				self._cache[cache_key] = elem

				if grid_dependent:
					cache_key_output = ('output', elem.output_grid)
					if wavelength_dependent:
						cache_key_output += (wavelength_key, )

					self._cache[cache_key_output] = elem

				return elem
				
			def forward(self, wavefront, *args, **kwargs):
				return self.get_instance(input_grid=wavefront.electric_field.grid, wavelength=wavefront.wavelength).forward(wavefront, *args, **kwargs)
			
			def backward(self, wavefront, *args, **kwargs):
				return self.get_instance(output_grid=wavefront.electric_field.grid, wavelength=wavefront.wavelength).backward(wavefront, *args, **kwargs)
			
			def get_transformation_matrix_forward(self, input_grid, wavelength, *args, **kwargs):
				return self.get_instance(input_grid=input_grid, wavelength=wavelength).get_transformation_matrix_forward(input_grid, wavelength, *args, **kwargs)
			
			def get_transformation_matrix_backward(self, output_grid, wavelength, *args, **kwargs):
				return self.get_instance(output_grid=output_grid, wavelength=wavelength).get_transformation_matrix_backward(output_grid, wavelength, *args, **kwargs)

		return AgnosticOpticalElement
	return decorator

def make_polychromatic(evaluated_arguments=None, num_in_cache=50):
	'''Make a polychromatic optical element from a monochromatic one.

	This decorator is deprecated and will be removed in a future version. New code 
	should use its successor :func:`make_agnostic_optical_element`, which can also
	handle input_grid dependencies.

	Parameters
	----------
	evaluated_arguments : list of strings or None
		A list of parameters that can be a function of wavelength. If this is None,
		the original optical element can already handle all wavelengths.
	num_in_cache : int
		The maximum size of the internal cache.
	
	Returns
	-------
	class
		An optical element class that can handle all wavelengths.
	'''
	def decorator(optical_element):
		class PolychromaticOpticalElement(OpticalElement):
			def __init__(self, *args, **kwargs):
				self.wavelengths = []
				self.monochromatic_optical_elements = []
				self.monochromatic_args = args
				self.monochromatic_kwargs = kwargs

				if evaluated_arguments is not None:
					init = optical_element.__init__
					if hasattr(inspect, 'signature'):
						# Python 3
						monochromatic_arg_names = list(inspect.signature(init).parameters.keys())[1:]
					else:
						# Python 2
						monochromatic_arg_names = inspect.getargspec(init).args
					
					self.evaluate_arg = [m in evaluated_arguments for m in monochromatic_arg_names]

			def get_instance(self, input_grid, wavelength):
				if self.wavelengths:
					i = np.argmin(np.abs(wavelength - np.array(self.wavelengths)))
					wavelength_closest = self.wavelengths[i]

					delta_wavelength = np.abs(wavelength - wavelength_closest)
					if (delta_wavelength / wavelength) < 1e-6:
						return self.monochromatic_optical_elements[i]
				
				if evaluated_arguments is not None:
					args = list(self.monochromatic_args)
					kwargs = dict(self.monochromatic_kwargs)

					for i, (arg, ev) in enumerate(zip(args, self.evaluate_arg)):
						if ev and callable(arg):
							args[i] = arg(wavelength)
					
					for key, val in kwargs.items():
						if key in evaluated_arguments and callable(val):
							kwargs[key] = val(wavelength)
					
					elem = optical_element(*args, wavelength=wavelength, **kwargs)
				else:
					elem = optical_element(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)
				
				self.wavelengths.append(wavelength)
				self.monochromatic_optical_elements.append(elem)

				if len(self.wavelengths) > num_in_cache:
					self.wavelengths.pop(0)
					self.monochromatic_optical_elements.pop(0)
				
				return elem
			
			def forward(self, wavefront, *args, **kwargs):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).forward(wavefront, *args, **kwargs)
			
			def backward(self, wavefront, *args, **kwargs):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).backward(wavefront, *args, **kwargs)
			
			def get_transformation_matrix_forward(self, input_grid, wavelength, *args, **kwargs):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_forward(input_grid, wavelength, *args, **kwargs)
			
			def get_transformation_matrix_backward(self, input_grid, wavelength, *args, **kwargs):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_backward(input_grid, wavelength, *args, **kwargs)
		
		return PolychromaticOpticalElement
	return decorator

class OpticalSystem(OpticalElement):
	'''An linear path of optical elements.

	Parameters
	----------
	optical_elements : list of OpticalElement
		The optical elements in the order that the wavefront propagates.
	'''
	def __init__(self, optical_elements):
		self.optical_elements = optical_elements

	def forward(self, wavefront):
		'''Propagate a wavefront forward through the optical system.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		wf = wavefront.copy()

		for optical_element in self.optical_elements:
			wf = optical_element.forward(wf)
		
		return wf
	
	def backward(self, wavefront):
		'''Propagate a wavefront backward through the optical system.

		This will be implemented by the derived class.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		wf = wavefront.copy()

		for optical_element in reversed(self.optical_elements):
			wf = optical_element.backward(wf)
		
		return wf
	
	def get_transformation_matrix_forward(self, wavelength=1):
		'''Calculate the forward linear transformation matrix.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		matrix = np.array(1)

		for optical_element in self.optical_elements:
			matrix = np.dot(optical_element.get_transformation_matrix_forward(wavelength), matrix)
		
		return matrix
	
	def get_transformation_matrix_backward(self, wavelength=1):
		'''Calculate the forward linear transformation matrix.

		Parameters
		----------
		wavelength : scalar
			The wavelength for which the transformation matrix will be calculated.
		
		Returns
		-------
		ndarray
			The full propagation matrix.
		'''
		matrix = np.array(1)

		for optical_element in reversed(self.optical_elements):
			matrix = np.dot(optical_element.get_transformation_matrix_backward(wavelength), matrix)
		
		return matrix
	
	@property
	def optical_elements(self):
		'''The list of optical elements contained in this optical system.
		'''
		return self._optical_elements

	@optical_elements.setter
	def optical_elements(self, optical_elements):
		self._optical_elements = list(optical_elements)
