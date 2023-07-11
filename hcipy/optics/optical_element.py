import numpy as np
import inspect
import collections
import itertools

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

class InstanceData(object):
    '''An object that stores data related to a single instance of input_grid, output_grid and wavelength.

    Parameters
    ----------
    input_grid : Grid or None
        The input grid.
    output_grid : Grid or None
        The output grid.
    wavelength : scalar or None
        The wavelength.
    '''
    def __init__(self, input_grid, output_grid, wavelength):
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.wavelength = wavelength

def _get_function_parameters(func):
    '''Get the names of the parameters for a function for both Python2 and Python3.

    Parameters
    ----------
    func : function
        The function to get the parameter names for.

    Returns
    -------
    list of str
        The list with the name for each of the parameters of the given function.
    '''
    if hasattr(inspect, 'signature'):
        # Python 3
        return list(inspect.signature(func).parameters.keys())
    else:
        # Python 2
        return inspect.getargspec(func).args

INPUT_GRID_DEPENDENT = 1
OUTPUT_GRID_DEPENDENT = 2
WAVELENGTH_DEPENDENT = 4

class AgnosticOpticalElement(OpticalElement):
    '''Base class for optical elements that require additional processing/caching for supporting different grids or wavelengths.

    This class is mant to simplify the creation of agnostic optical elements. When you have an
    optical element that explicitly needs an input grid and/or wavelength on initialization, you can
    use this class to simplify making it accept all input/output grids and/or wavelengths.

    Instances are created by a function `make_instance` that derived classes should use to evaluate properties
    on the given input_grid, output_grid and wavelength. These properties are stored in an internal cache, and
    are reused for as long as they reside in the cache. Any function using forward or backward propagation will
    have access to the cached instance data. All caching and retrieving of cached data is made invisible to the
    user.

    As instanced data can take a lot of memory, at most `max_in_cache` instances will be held in the cache at
    the time. If an additional instance is requested, the oldest instance will be thrown away, and the new
    instance will be put in the cache instead.

    Parameters
    ----------
    grid_dependent : boolean
        If the instances should be separated by grid. Separate instances will be made if their grids change
        between invocations.
    wavelength_dependent : boolean
        If the instances should be separated by grid.Separate instances will be made if wavelength changes
        between invocations.
    max_in_cache : int
        The maximum size of the internal cache for optical elements. Reduce this if the cache is using
        too much memory, increase if there are a lot of cache misses.
    '''
    def __init__(self, grid_dependent=True, wavelength_dependent=True, max_in_cache=11):
        self._grid_dependent = grid_dependent
        self._wavelength_dependent = wavelength_dependent
        self._max_in_cache = max_in_cache

        self.clear_cache()

    def _get_cache_keys(self, input_grid, output_grid, wavelength):
        '''Get all keys for the instance cache.

        Parameters
        ----------
        input_grid : Grid or None
            The input grid.
        output_grid : Grid or None
            The output grid.
        wavelength : scalar or None
            The wavelength.

        Returns
        -------
        list
            The list of cache keys.
        '''
        key_parts = []

        if self._grid_dependent:
            if input_grid is None:
                if output_grid is None:
                    raise ValueError('Grid dependent, but no grids are given for lookup.')

                key_parts.append([(None, hash(output_grid))])
            elif output_grid is None:
                key_parts.append([(hash(input_grid), None)])
            else:
                key_parts.append([(hash(input_grid), hash(output_grid)), (hash(input_grid), None), (None, hash(output_grid))])
        else:
            key_parts.append([(None, None)])

        if self._wavelength_dependent:
            if wavelength is None:
                raise ValueError('Wavelength dependent, but no wavelength is given for lookup.')

            wavelength_key = int(np.round(np.log(wavelength) / np.log(1 + 1e-9)))
            key_parts.append([(wavelength_key,)])
        else:
            key_parts.append([(None,)])

        keys = []
        for parts in itertools.product(*key_parts):
            key = ()
            for p in parts:
                key += p
            keys.append(key)

        return keys

    def clear_cache(self):
        '''Clear the instance cache.

        This function should be called if agnostic data, that was used
        to create instance data, was changed by the user. Clearing the
        cache ensures that the propagations are always performed using
        up-to-date arguments.
        '''
        self._instance_data_cache = collections.OrderedDict()
        self._num_in_cache = 0

    def _add_to_cache(self, instance_data, cache_keys=None):
        if self._num_in_cache == self._max_in_cache:
            # Remove last added item
            key, value = self._instance_data_cache.popitem(False)

            # Remove all copies of that item as well
            old_cache_keys = self._get_cache_keys(value.input_grid, value.output_grid, value.wavelength)
            for i in range(len(old_cache_keys) - 1):
                self._instance_data_cache.popitem(False)

            # Update number of instance data items
            self._num_in_cache -= 1

        # Calculate cache keys
        if cache_keys is None:
            cache_keys = self._get_cache_keys(instance_data.input_grid, instance_data.output_grid, instance_data.wavelength)

        # Add instance data under each of these keys
        for cache_key in cache_keys:
            self._instance_data_cache[cache_key] = instance_data

        # Update number of instance data items
        self._num_in_cache += 1

    def _get_parameter_signature(self, parameter):
        '''Guess the signature of a given parameter.

        A parameter can be a function of (input_grid, output_grid, wavelength)
        or a subset of this. This function will try to guess what (sub)set should
        be used to evaluate this parameter.

        Parameters
        ----------
        parameter : anything
            The parameter for which to guess the signature.

        Returns
        -------
        int
            A bit-map indicating the best-guess signature.
        '''
        if not callable(parameter):
            # Parameter is not callable, so no evaluation can be done.
            return 0

        param_parameters = _get_function_parameters(parameter)

        if len(param_parameters) == 1:
            if self._grid_dependent and self._wavelength_dependent:
                # Need to choose between the two
                if 'grid' in param_parameters[0]:
                    return INPUT_GRID_DEPENDENT
                elif 'lam' in param_parameters[0] or 'wave' in param_parameters[0] or 'wvl' in param_parameters[0]:
                    return WAVELENGTH_DEPENDENT
                return WAVELENGTH_DEPENDENT
            elif self._grid_dependent:
                # Only grid dependent
                return INPUT_GRID_DEPENDENT
            elif self._wavelength_dependent:
                # Only wavelength dependent
                return WAVELENGTH_DEPENDENT
            else:
                return 0
        elif len(param_parameters) == 2:
            if not self._wavelength_dependent:
                return INPUT_GRID_DEPENDENT | OUTPUT_GRID_DEPENDENT
            else:
                return INPUT_GRID_DEPENDENT | WAVELENGTH_DEPENDENT
        elif len(param_parameters) == 3:
            return INPUT_GRID_DEPENDENT | OUTPUT_GRID_DEPENDENT | WAVELENGTH_DEPENDENT
        else:
            return 0

    def evaluate_parameter(self, parameter, input_grid, output_grid, wavelength):
        '''Evaluate the parameter as function of (input_grid, output_grid, wavelength).

        The parameter can be a function of all or a subset of these parameters. This function
        will try to guess the function signature and attempt evaluation of the given function.

        Parameters
        ----------
        parameter : anything
            The parameter to evaluate.
        input_grid : Grid or None
            The input grid.
        output_grid : Grid or None
            The output grid.
        wavelength : scalar or None
            The wavelength.

        Returns
        -------
        any type
            The evaluated parameter.

        Raises
        ------
        RuntimeError
            If the function could not be evaluated using the best-guess signature.
        '''
        signature = self._get_parameter_signature(parameter)

        if not signature:
            return parameter

        args = []
        if signature & INPUT_GRID_DEPENDENT:
            if input_grid is None:
                input_grid = self.get_input_grid(output_grid, wavelength)

            args.append(input_grid)
        if signature & OUTPUT_GRID_DEPENDENT:
            if output_grid is None:
                output_grid = self.get_output_grid(input_grid, wavelength)

            args.append(output_grid)
        if signature & WAVELENGTH_DEPENDENT:
            args.append(wavelength)

        try:
            return parameter(*args)
        except Exception:
            raise RuntimeError('Parameter could not be evaluated.')

    def construct_function(self, function, *args, **kwargs):
        '''Construct a function based on the given function and arguments.

        The arguments can be (input_grid, output_grid, wavelength) or any subset of this.
        The returned function has parameters which encompass all given parameters.

        This function is especially usefull for creating properties that depend on
        parameters that depend on input_grid, output_grid and/or wavelength.

        Parameters
        ----------
        function : function
            The function on which to base the returned function.
        *args : anything
            The arguments for the given function.
        **kwargs : anything
            The keyword arguments for the given function.

        Returns
        -------
        function
            The constructed function.

        Raises
        ------
        RuntimeError
            If the signature of one of the parameter is not recognized.
        '''
        # Evaluate function with parameters that can be grid and/or wavelength dependent
        # This returns a function with arguments chosen to fit whether the whole thing is grid and/or wavelength dependent.
        signature = 0

        for arg in args:
            signature |= self._get_parameter_signature(arg)

        for kwarg in kwargs.values():
            signature |= self._get_parameter_signature(kwarg)

        # Check original function for direct parameter dependencies
        function_params = _get_function_parameters(function)

        if 'input_grid' in function_params:
            kwargs['input_grid'] = lambda input_grid: input_grid
            signature |= INPUT_GRID_DEPENDENT

        if 'output_grid' in function_params:
            kwargs['output_grid'] = lambda output_grid: output_grid
            signature |= OUTPUT_GRID_DEPENDENT

        if 'wavelength' in function_params:
            kwargs['wavelength'] = lambda wavelength: wavelength
            signature |= WAVELENGTH_DEPENDENT

        # If parameters are not a function of anything, evaluate the function now and return the result
        if signature == 0:
            return function(*args, **kwargs)

        # Put all args in kwargs, but ignore input_grid, output_grid and wavelength
        index = 0
        for param_name in function_params:
            if param_name in ['input_grid', 'output_grid', 'wavelength']:
                continue

            kwargs[param_name] = args[index]
            index += 1

            if len(args) == index:
                break

        # Create the returned function
        if signature == (INPUT_GRID_DEPENDENT | OUTPUT_GRID_DEPENDENT | WAVELENGTH_DEPENDENT):
            def func(input_grid, output_grid, wavelength):
                evaluated_kwargs = {}
                for key, val in kwargs.items():
                    evaluated_kwargs[key] = self.evaluate_parameter(val, input_grid, output_grid, wavelength)

                return function(**evaluated_kwargs)
        elif signature == (INPUT_GRID_DEPENDENT | OUTPUT_GRID_DEPENDENT):
            def func(input_grid, output_grid):
                evaluated_kwargs = {}
                for key, val in kwargs.items():
                    evaluated_kwargs[key] = self.evaluate_parameter(val, input_grid, output_grid, None)

                return function(**evaluated_kwargs)
        elif signature == (INPUT_GRID_DEPENDENT | WAVELENGTH_DEPENDENT):
            def func(input_grid, wavelength):
                evaluated_kwargs = {}
                for key, val in kwargs.items():
                    evaluated_kwargs[key] = self.evaluate_parameter(val, input_grid, None, wavelength)

                return function(**evaluated_kwargs)
        elif signature == INPUT_GRID_DEPENDENT:
            def func(input_grid):
                evaluated_kwargs = {}
                for key, val in kwargs.items():
                    evaluated_kwargs[key] = self.evaluate_parameter(val, input_grid, None, None)

                return function(**evaluated_kwargs)
        elif signature == WAVELENGTH_DEPENDENT:
            def func(wavelength):
                evaluated_kwargs = {}
                for key, val in kwargs.items():
                    evaluated_kwargs[key] = self.evaluate_parameter(val, None, None, wavelength)

                return function(**evaluated_kwargs)
        else:
            raise RuntimeError('Signature %d was not recognized.' % signature)

        return func

    def get_instance_data(self, input_grid, output_grid, wavelength):
        '''Get the InstanceData object corresponding to the given grids and wavelength.

        If no InstanceData can be found in the internal cache, a new instance will be
        constructed.

        Parameters
        ----------
        input_grid : Grid or None
            The input grid.
        output_grid : Grid or None
            The output grid.
        wavelength : scalar or None
            The wavelength.
        '''
        cache_keys = self._get_cache_keys(input_grid, output_grid, wavelength)

        for cache_key in cache_keys:
            if cache_key in self._instance_data_cache:
                instance_data = self._instance_data_cache[cache_key]
                break
        else:
            # Try to guess input and output grid.
            if input_grid is None:
                input_grid = self.get_input_grid(output_grid, wavelength)

            if output_grid is None:
                output_grid = self.get_output_grid(input_grid, wavelength)

            # Recalculate cache keys and try again.
            cache_keys = self._get_cache_keys(input_grid, output_grid, wavelength)

            for cache_key in cache_keys:
                if cache_key in self._instance_data_cache:
                    instance_data = self._instance_data_cache[cache_key]
                    break
            else:
                # Item does not yet exist. Create instanceData element
                instance_data = InstanceData(input_grid, output_grid, wavelength)
                self.make_instance(instance_data, input_grid, output_grid, wavelength)

                # Add instance data to cache.
                self._add_to_cache(instance_data, cache_keys)

        return instance_data

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        '''Make an instance for this specific input_grid, output_grid, wavelength.

        This function is intended to be implemented by a derived class. Any properties
        evaluated for the instance can be stored into the instance_data object which
        is stored in the internal cache.

        Parameters
        ----------
        instance_data : InstanceData
            An object storing all data for this instance. This object can be modified this function.
        input_grid : Grid or None
            The input grid.
        output_grid : Grid or None
            The output grid.
        wavelength : scalar or None
            The wavelength.
        '''
        pass

    def get_input_grid(self, output_grid, wavelength):
        '''Calculate a best guess for the input grid given an output grid and wavelength.

        This function is intended to be implemented by a derived class.

        Parameters
        ----------
        output_grid : Grid
            The output grid.
        wavelength : scalar
            The wavelength.

        Returns
        -------
        Grid
            The best-guess input grid based on the given output grid and wavelength.
        '''
        return None

    def get_output_grid(self, input_grid, wavelength):
        '''Calculate a best guess for the output grid given an input grid and wavelength.

        This function is intended to be implemented by a derived class.

        Parameters
        ----------
        input_grid : Grid
            The input grid.
        wavelength : scalar
            The wavelength.

        Returns
        -------
        Grid
            The best-guess output grid based on the given input grid and wavelength.
        '''

        return None

    def __getattr__(self, name):
        '''A redirect for instance data.

        Any attribute that exists only in an instance can be accessed by this redirect.

        Parameters
        ----------
        name : str
            The name of the requested attribute.

        Returns
        -------
        function
            The attribute that can be evaluated on a specific input, output grid and wavelength.
        '''
        def attribute(input_grid=None, output_grid=None, wavelength=None):
            instance_data = self.get_instance_data(input_grid, output_grid, wavelength)

            return getattr(instance_data, name)
        return attribute

    def __getstate__(self):
        '''Get the state of the optical element for pickle.

        Returns
        -------
        dict
            All contained variables.
        '''
        return self.__dict__

    def __setstate__(self, state):
        '''Set the state of the optical element for pickle.

        Parameters
        ----------
        state : dict
            The state of an optical element returned by a __getstate__().
        '''
        self.__dict__ = state

def make_agnostic_forward(forward):
    '''A decorator for a forward function on an AgnosticOpticalElement.

    Any derived class should use this decorator on any forward-type function. This
    allows the function to have access to instance data, in addition to agnostic properties.

    Parameters
    ----------
    forward : function
        The modified forward-type function.

    Returns
    -------
    function
        The new forward-type function that calls the old function with added instance data.
    '''
    def res(self, wavefront, *args, **kwargs):
        # Look up instance data
        instance_data = self.get_instance_data(wavefront.grid, None, wavefront.wavelength)

        return forward(self, instance_data, wavefront, *args, **kwargs)
    return res

def make_agnostic_backward(backward):
    '''A decorator for a backward function on an AgnosticOpticalElement.

    Any derived class should use this decorator on any backward-type function. This
    allows the function to have access to instance data, in addition to agnostic properties.

    Parameters
    ----------
    backward : function
        The modified backward-type function.

    Returns
    -------
    function
        The new backward-type function that calls the old function with added instance data.
    '''
    def res(self, wavefront, *args, **kwargs):
        # Look up instance data
        instance_data = self.get_instance_data(None, wavefront.grid, wavefront.wavelength)

        return backward(self, instance_data, wavefront, *args, **kwargs)
    return res

def make_agnostic_optical_element(grid_dependent_arguments=None, wavelength_dependent_arguments=None, num_in_cache=50):  # pragma: no cover
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
    import warnings
    warnings.warn('This function will be removed in a future release. Please switch to a AgnosticOpticalElement class instead.', DeprecationWarning)

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
        wf = wavefront

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
        wf = wavefront

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
