import numpy as np

from ..optics import Wavefront, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from ..field import Field
from ..fourier import make_fourier_transform

class FraunhoferPropagator(AgnosticOpticalElement):
    '''A monochromatic perfect lens propagator.

        This implements the propagation of a wavefront through a perfect lens. The wavefront
        is assumed to be exactly in the front focal plane of the lens and is propagated to the
        back focal plane. The implementation follows [Goodman2005]_.

        .. [Goodman2005] Goodman, J.W., 2005 Introduction to Fourier optics. Roberts and Company Publishers.

        Parameters
        ----------
        input_grid : Grid
            The grid on which the incoming wavefront is defined.
        output_grid : Grid
            The grid on which the outgoing wavefront is to be evaluated.
        focal_length : scalar
            The focal length of the lens system.
    '''
    def __init__(self, input_grid, output_grid, focal_length=1):
        self._input_grid = input_grid
        self._output_grid = output_grid
        self._focal_length = focal_length

        AgnosticOpticalElement.__init__(self, grid_dependent=True, wavelength_dependent=True)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        focal_length = self.evaluate_parameter(self.focal_length, input_grid, output_grid, wavelength)

        instance_data.uv_grid = output_grid.scaled(2 * np.pi / (focal_length * wavelength))
        instance_data.fourier_transform = make_fourier_transform(input_grid, instance_data.uv_grid)

        instance_data.norm_factor = 1 / (1j * focal_length * wavelength)

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, focal_length):
        self._focal_length = focal_length

        self.clear_cache()

    def get_input_grid(self, output_grid, wavelength):
        return self._input_grid

    def get_output_grid(self, input_grid, wavelength):
        return self._output_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        '''Propagate a wavefront forward through the lens.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        U_new = instance_data.fourier_transform.forward(wavefront.electric_field) * instance_data.norm_factor
        return Wavefront(Field(U_new, instance_data.output_grid), wavefront.wavelength, wavefront.input_stokes_vector)

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        '''Propagate a wavefront backward through the lens.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The wavefront after the propagation.
        '''
        U_new = instance_data.fourier_transform.backward(wavefront.electric_field) / instance_data.norm_factor
        return Wavefront(Field(U_new, instance_data.input_grid), wavefront.wavelength, wavefront.input_stokes_vector)
