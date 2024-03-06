from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..aperture import make_circular_aperture
from ..optics import SurfaceApodizer, Apodizer, TipTiltMirror
from ..field import make_pupil_grid, Field, CartesianGrid, UnstructuredCoords
from ..field import evaluate_supersampled

import numpy as np

class ModulatedPyramidWavefrontSensorOptics(WavefrontSensorOptics):
    '''The optical elements for a modulated pyramid wavefront sensor.

    This class supports both a slow (default) and a fast modulation method, chosen by the
    fast_modulation_method parameters. The fast method shifts the pyramid surface instead of
    applying tilts to the incoming wavefront. While the fast modulation method can be up to
    twice as fast, it needs to precompute all pyramid surfaces and will use up more memory.

    Parameters
    ----------
    pyramid_wavefront_sensor : WavefrontSensorOptics
        The pyramid wavefront sensor optics that are used.
    modulation : scalar
        The modulation radius in radians.
    num_steps : int
        The number of steps per modulation cycle.
    fast_modulation_method : boolean
        If True the fast propagation method will be used. Default is False.
    '''
    def __init__(self, pyramid_wavefront_sensor, modulation, num_steps=12, fast_modulation_method=False):
        self.modulation = modulation
        self.pyramid_wavefront_sensor = pyramid_wavefront_sensor
        self.tip_tilt_mirror = TipTiltMirror(self.pyramid_wavefront_sensor.input_grid)
        self.focal_grid = self.pyramid_wavefront_sensor.focal_grid

        self.pupil_to_focal = self.pyramid_wavefront_sensor.pupil_to_focal
        self.focal_to_pupil = self.pyramid_wavefront_sensor.focal_to_pupil
        self._fast_modulation_method = fast_modulation_method

        # Calculate the modulation positions
        theta = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
        x_modulation = modulation / 2 * np.cos(theta)
        y_modulation = modulation / 2 * np.sin(theta)

        self.modulation_positions = CartesianGrid(UnstructuredCoords((x_modulation, y_modulation)))

        separation = self.pyramid_wavefront_sensor._separation
        refractive_index = self.pyramid_wavefront_sensor._refractive_index
        wavelength_0 = self.pyramid_wavefront_sensor._wavelength_0

        # Create all the shifted pyramid surfaces
        def surface_function(temp_grid):
            return Field(-separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(temp_grid.x) + np.abs(temp_grid.y)), temp_grid)

        self._pyramid_apodizers = []
        for p in self.modulation_positions:
            # The factor of two is to compensate for the factor of 1/2 of the mirror surface.
            shifted_grid = self.focal_grid.shifted(2 * p)

            pyramid_surface = evaluate_supersampled(surface_function, shifted_grid, 4)
            pyramid_apodizer = SurfaceApodizer(Field(pyramid_surface, self.focal_grid), refractive_index)

            self._pyramid_apodizers.append(pyramid_apodizer)

    def forward(self, wavefront):
        '''Propagates a wavefront through the modulated pyramid wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf_modulated : list
            A list of wavefronts for each modulation position.
        '''
        if self._fast_modulation_method:
            wf_foc = self.pyramid_wavefront_sensor.spatial_filter(self.pupil_to_focal(wavefront))
            wf_modulated = [self.focal_to_pupil(apod(wf_foc)) for apod in self._pyramid_apodizers]
        else:
            wf_modulated = []

            for point in self.modulation_positions.points:
                self.tip_tilt_mirror.actuators = point
                modulated_wavefront = self.tip_tilt_mirror.forward(wavefront)

                wf_modulated.append(self.pyramid_wavefront_sensor.forward(modulated_wavefront))

        return wf_modulated

    def backward(self, wavefront):
        raise RuntimeError('This is a non-physical operation.')

class PyramidWavefrontSensorOptics(WavefrontSensorOptics):
    '''The optical elements for a pyramid wavefront sensor.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the input wavefront is defined.
    output_grid : Grid
        The grid on which the output wavefront is defined.
    separation : scalar
        The separation between the pupils. The default takes the input grid extent as separation.
    pupil_diameter : scalar
        The diameter of the aperture.
    wavelength_0 : scalar
        The reference wavelength that determines the physical scales.
    q : scalar
        The focal plane oversampling coefficient. The default uses the minimal required sampling.
    num_airy : scalar
        The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
    refractive_index : callable
        A callable that returns the refractive index as function of wavelength.
        The default is a refractive index of 1.5.
    '''
    def __init__(self, input_grid, output_grid, separation=None, pupil_diameter=None, wavelength_0=1, q=None, num_airy=None, refractive_index=lambda x: 1.5):
        if not input_grid.is_regular:
            raise ValueError('The input grid must be a regular grid.')

        self.input_grid = input_grid
        self.output_grid = output_grid

        if pupil_diameter is None:
            pupil_diameter = np.max(input_grid.delta * input_grid.shape)

        if separation is None:
            separation = pupil_diameter

        # Create the intermediate focal grid
        # Oversampling is necessary to see all frequencies in the output wavefront sensor plane
        # and we require at least 2 pixels per spatial resolution element for the default case.
        qmin = max((output_grid.delta * output_grid.dims) / (input_grid.delta * input_grid.dims))
        qmin = np.ceil(max(qmin, 2))

        if q is None:
            q = qmin
        elif q < qmin:
            raise ValueError('The requested focal plane sampling is too low to sufficiently sample the wavefront sensor output.')

        if num_airy is None:
            self.num_airy = np.max(input_grid.shape) / 2
        else:
            self.num_airy = num_airy

        num_pixels = 2 * int(self.num_airy * q)
        spatial_resolution = wavelength_0 / pupil_diameter
        self.focal_grid = make_pupil_grid(num_pixels, 2 * spatial_resolution * self.num_airy)

        self._separation = separation
        self._refractive_index = refractive_index
        self._wavelength_0 = wavelength_0

        # Make all the optical elements
        self.spatial_filter = Apodizer(make_circular_aperture(2 * self.num_airy * wavelength_0 / pupil_diameter)(self.focal_grid))
        pyramid_surface = -separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(self.focal_grid.x) + np.abs(self.focal_grid.y))
        self.pyramid = SurfaceApodizer(Field(pyramid_surface, self.focal_grid), refractive_index)

        # Make the propagators
        self.pupil_to_focal = FraunhoferPropagator(input_grid, self.focal_grid)
        self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid)

    def forward(self, wavefront):
        '''Propagates a wavefront through the pyramid wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf_wfs : Wavefront
            The output wavefront.
        '''
        wf_focus = self.pupil_to_focal.forward(wavefront)
        wf_pyramid = self.pyramid.forward(self.spatial_filter.forward(wf_focus))
        wf_wfs = self.focal_to_pupil.forward(wf_pyramid)

        return wf_wfs

    def backward(self, wavefront):
        '''Propagates a wavefront backwards through the pyramid wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf_pupil : Wavefront
            The output wavefront.
        '''
        wf_focus = self.focal_to_pupil.backward(wavefront)
        wf_pyramid = self.pyramid.backward(self.spatial_filter.backward(wf_focus))
        wf_pupil = self.pupil_to_focal.backward(wf_pyramid)

        return wf_pupil

class PyramidWavefrontSensorEstimator(WavefrontSensorEstimator):
    '''Estimates the wavefront slopes from pyramid wavefront sensor images.

    Parameters
    ----------
    aperture : function
        A function which mask the pupils for the normalized differences.
    output_grid : Grid
        The grid on which the output of a pyramid wavefront sensor is sampled.

    Attributes
    ----------
    measurement_grid : Grid
        The grid on which the normalized differences are defined.
    pupil_mask : array_like
        A mask for the normalized differences.
    num_measurements : int
        The number of pixels in the output vector.
    '''
    def __init__(self, aperture):
        self.pupil_mask = aperture
        self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

    def estimate(self, images):
        '''A function which estimates the wavefront slope from a pyramid image.

        Parameters
        ----------
        images - list
            A list of scalar intensity fields containing pyramid wavefront sensor images.

        Returns
        -------
        res - Field
            A field with wavefront sensor slopes.
        '''
        import warnings
        warnings.warn("This function does not work as expected and will be changed in a future update.", RuntimeWarning)

        image = images.shaped
        sub_shape = image.grid.shape // 2

        # Subpupils
        I_a = image[:sub_shape[0], :sub_shape[1]]
        I_b = image[sub_shape[0]:2 * sub_shape[0], :sub_shape[1]]
        I_c = image[sub_shape[0]:2 * sub_shape[0], sub_shape[1]:2 * sub_shape[1]]
        I_d = image[:sub_shape[0], sub_shape[1]:2 * sub_shape[1]]

        norm = I_a + I_b + I_c + I_d
        inv_norm = np.zeros_like(norm)
        inv_norm[norm != 0] = 1 / norm[norm != 0]

        I_x = (I_a + I_b - I_c - I_d) * inv_norm
        I_y = (I_a - I_b - I_c + I_d) * inv_norm

        I_x = I_x.ravel()[self.pupil_mask > 0]
        I_y = I_y.ravel()[self.pupil_mask > 0]

        res = Field([I_x, I_y], self.pupil_mask.grid)
        return res
