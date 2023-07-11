import numpy as np
import numexpr as ne

from ..optics import Wavefront, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from ..field import Field, evaluate_supersampled
from ..fourier import FastFourierTransform, make_fft_grid, FourierFilter

class FresnelPropagator(AgnosticOpticalElement):
    '''The monochromatic Fresnel propagator for scalar fields.

    The Fresnel propagator is implemented as described in [Goodman2005]_.

    .. [Goodman2005] Goodman, J.W., 2005 Introduction to Fourier optics. Roberts and Company Publishers.

    Parameters
    ----------
    input_grid : anything
        This argument is ignored. The input grid is taken from the incoming wavefront.
    distance : scalar
        The distance to propagate
    num_oversampling : int
        The number of times the transfer function is oversampled. Default is 2.
    zero_padding : scalar
        The amount of zero padding to use for the propagation. A value of one means no
        zero padding is applied. Default is 2.
    refractive_index : scalar
        The refractive index of the medium that the wavefront is propagating in.

    Raises
    ------
    ValueError
        If the `input_grid` is not regular and Cartesian.
    '''
    def __init__(self, input_grid, distance, num_oversampling=2, zero_padding=2, refractive_index=1):
        self._distance = distance

        self._num_oversampling = num_oversampling
        self._zero_padding = zero_padding
        self._refractive_index = refractive_index

        AgnosticOpticalElement.__init__(self, grid_dependent=True, wavelength_dependent=True)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        if not input_grid.is_regular or not input_grid.is_('cartesian'):
            raise ValueError('The input grid must be a regular, Cartesian grid.')

        k = 2 * np.pi / wavelength * self.evaluate_parameter(self.refractive_index, input_grid, output_grid, wavelength)
        L_max = np.max(input_grid.dims * input_grid.delta)

        if np.any(input_grid.delta < wavelength * abs(self.distance) / L_max):
            def transfer_function(fourier_grid):
                enlarged_grid = make_fft_grid(fourier_grid)
                fft_upscale = FastFourierTransform(enlarged_grid)

                def impulse_response(grid):
                    x = grid.x
                    y = grid.y
                    phase_factor = np.exp(1j * k * self.distance) / (1j * wavelength * self.distance)

                    variables = {'alpha': 0.5j * k / self.distance, 'x': x, 'y': y, 'phase_factor': phase_factor}
                    ir = ne.evaluate('phase_factor * exp(alpha * (x * x + y * y))', local_dict=variables)

                    return Field(ir, grid)

                impulse_response = evaluate_supersampled(impulse_response, enlarged_grid, self.num_oversampling)

                return fft_upscale.forward(impulse_response)
        else:
            def transfer_function_native(fourier_grid):
                x = fourier_grid.x
                y = fourier_grid.y
                phase_factor = np.exp(1j * k * self.distance)

                variables = {'alpha': -0.5j * self.distance / k, 'x': x, 'y': y, 'phase_factor': phase_factor}
                tf = ne.evaluate('phase_factor * exp(alpha * (x * x + y * y))', local_dict=variables)

                return Field(tf, fourier_grid)

            def transfer_function(fourier_grid):
                return evaluate_supersampled(transfer_function_native, fourier_grid, self.num_oversampling)

        instance_data.fourier_filter = FourierFilter(input_grid, transfer_function, q=self.zero_padding)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

        self.clear_cache()

    @property
    def num_oversampling(self):
        return self._num_oversampling

    @num_oversampling.setter
    def num_oversampling(self, num_oversampling):
        self._num_oversampling = num_oversampling

        self.clear_cache()

    @property
    def zero_padding(self):
        return self._zero_padding

    @zero_padding.setter
    def zero_padding(self, zero_padding):
        self._zero_padding = zero_padding

        self.clear_cache()

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, refractive_index):
        self._refractive_index = refractive_index

        self.clear_cache()

    def get_input_grid(self, output_grid, wavelength):
        return output_grid

    def get_output_grid(self, input_grid, wavelength):
        return input_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        '''Propagate a wavefront forward by a certain distance.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        filtered = instance_data.fourier_filter.forward(wavefront.electric_field)

        return Wavefront(filtered, wavefront.wavelength, wavefront.input_stokes_vector)

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        '''Propagate a wavefront backward by a certain distance.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        filtered = instance_data.fourier_filter.backward(wavefront.electric_field)

        return Wavefront(filtered, wavefront.wavelength, wavefront.input_stokes_vector)
