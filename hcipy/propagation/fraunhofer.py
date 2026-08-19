from ..optics import Wavefront, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from ..field import Field
from ..fourier import make_fourier_transform

import math

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

        instance_data.uv_grid = output_grid.scaled(2 * math.pi / (focal_length * wavelength))
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

    def closest_fft_wavelength(self, target_wavelength):
        '''Find the wavelength closest to `target_wavelength` for which the
        propagation could be performed with a Fast Fourier Transform.

        .. note::
            A Fast Fourier Transform is not necessarily used for the returned
            wavelength; the transform to use is decided by
            :func:`~hcipy.fourier.make_fourier_transform`. The returned
            wavelength only guarantees that an FFT is a valid option, and will
            be used if it is estimated to be the fastest.

        Parameters
        ----------
        target_wavelength : scalar or array_like
            The wavelength(s) of interest.

        Returns
        -------
        Array
            The wavelength(s) closest to `target_wavelength` that can be
            propagated with an FFT.

        Raises
        ------
        ValueError
            If appropriate wavelengths could not be found.
        '''
        if not self._input_grid.is_regular or not self._output_grid.is_regular:
            raise ValueError('The input and output grid need to be regular for an FFT to be possible.')

        xp = self._input_grid.xp
        z = self._input_grid.delta * self._output_grid.delta

        eps = xp.finfo(z.dtype).eps * 1e2

        if not xp.all(xp.abs(z / z[0] - 1) < eps):
            raise ValueError('input_grid.delta * output_grid.delta needs to be equal for all axes for an FFT to be possible.')

        b = self.focal_length / z[0]

        # The zero-padding zp = b * wavelength needs to be an integer (and to
        # cover both grids, so at least the largest of their dims).
        zp_min = max(max(self._input_grid.dims), max(self._output_grid.dims))

        targets = xp.asarray(target_wavelength)
        zp = xp.maximum(xp.round(b * targets), zp_min)

        return zp / b
