import numpy as np
from ..util import large_poisson
from ..field import subsample_field, make_supersampled_grid

class Detector(object):
    '''Base class for a detector.

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector returns its images. These indicate
        the centers of the pixels.
    subsamping : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.

    Attributes
    ----------
    input_grid : Grid
        The grid that is expected as input.
    '''
    def __init__(self, detector_grid, subsamping=1):
        self.detector_grid = detector_grid
        self.subsamping = subsamping

        if subsamping > 1:
            self.input_grid = make_supersampled_grid(detector_grid, subsamping)
        else:
            self.input_grid = detector_grid

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        raise NotImplementedError()

    def read_out(self):
        '''Reads out the detector.

        No noise will be added to the image.

        Returns
        -------
        Field
            The final detector image.
        '''
        raise NotImplementedError()

    def __call__(self, wavefront, dt=1, weight=1):
        '''Integrate and read out the detector.

        This is a convenience function to avoid having to call two functions
        in quick succession.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.

        Returns
        -------
        Field
            The final detector image.
        '''
        self.integrate(wavefront, dt, weight)
        return self.read_out()

class NoiselessDetector(Detector):
    '''A detector without noise.

    This detector includes no noise effects at all. This can be used as a
    theoretically perfect detector.

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector returns its images. These indicate
        the centers of the pixels.
    subsamping : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, subsamping=1):
        Detector.__init__(self, detector_grid, subsamping)

        self.accumulated_charge = 0

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += power * dt * weight

    def read_out(self):
        '''Reads out the detector.

        No noise will be added to the image.

        Returns
        -------
        Field
            The final detector image.
        '''
        # Make sure not to overwrite output
        output_field = self.accumulated_charge.copy()

        # Reset detector
        self.accumulated_charge = 0

        return output_field

class NoisyDetector(Detector):
    '''A detector class that has some basic noise properties.

    A detector that includes the following noise sources: photon shot noise, dark current
    shot noise, flat field errors and read noise. These noise sources (except for photon shot
    noise) can be either a scalar (same for entire grid) or an array. For the flat field error
    specifically: if a scalar is given, a normal distributed flat field map (given scalar
    is standard deviation about 1) is generated that will be used - if an array is given, that
    array will be used, so it will NOT be a normal distributed flat field where every point in
    array gives the standard deviation. This allows the user to load a specific flat field map.

    The detector can also integrate supersampled wavefronts and subsample them to the correct
    grid size. This allows for the averaging of the power over one detector pixel.

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    dark_current_rate : scalar or array_like
        An array or scalar that gives dark current rate in counts per unit time for each point
        in the grid.
    read_noise : scalar or array_like
        An array or scalar that gives the rms read noise counts for each point in the grid.
    flat_field : scalar or array_like
        An array or scalar that gives the flat field error for each point in the
        grid. If a scalar is given, a random normal distributed flat field map (given scalar
        is standard deviation about 1) is generated that will be used.
    include_photon_noise : boolean
        Turns the photon noise on or off. Default: True.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, dark_current_rate=0, read_noise=0, flat_field=0, include_photon_noise=True, subsampling=1):
        Detector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = dark_current_rate
        self.read_noise = read_noise
        self.flat_field = flat_field
        self.include_photon_noise = include_photon_noise

    @property
    def flat_field(self):
        return self._flat_field

    @flat_field.setter
    def flat_field(self, flat_field):
        # If the flatfield parameters was a scalar, we will generate a flat field map that will
        # be constant for this object until flat_field is manually changed.
        if np.isscalar(flat_field):
            self._flat_field = np.random.normal(loc=1.0, scale=flat_field, size=self.detector_grid.size)
        else:
            self._flat_field = flat_field

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        The amount of power and dark current that the detector generates are calculated
        for a given integration time and weight.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def read_out(self):
        '''Reads out the detector.

        The read out operation of the detector. This means that, if applicable, first photon noise
        will be included to the power. Then, in this order, the flat field error and
        read-out noise are applied. After the read out the power is reset.

        Returns
        -------
        Field
            The final detector image.
        '''
        # Make sure not to overwrite output
        output_field = self.accumulated_charge.copy()

        # Adding photon noise.
        if self.include_photon_noise:
            output_field = large_poisson(output_field, thresh=1e6)

        # Adding flat field errors.
        output_field *= self.flat_field

        # Adding read-out noise.
        output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

        # Reset detector
        self.accumulated_charge = 0

        return output_field

class FrameCorrector(object):
    def correct(self, img):
        return img

class BasicFrameCorrector(FrameCorrector):
    def __init__(self, dark=0, flat_field=1):
        self.dark = dark
        self.flat_field = flat_field

    def correct(self, img):
        return (img - self.dark) / self.flat_field
