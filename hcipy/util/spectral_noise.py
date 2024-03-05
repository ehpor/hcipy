import numpy as np
import copy

from ..field import Field
from ..fourier import FastFourierTransform, MatrixFourierTransform

class SpectralNoiseFactory(object):
    def __init__(self, psd, output_grid):
        '''A factory class for spectral noise.

        Parameters
        ----------
        psd : Field generator
            The power spectral density of the noise.
        output_grid : Grid
            The grid on which to compute the noise.
        '''
        self.psd = psd
        self.output_grid = output_grid

    def make_random(self, seed=None):
        '''Make a single realization of the spectral noise.

        This function needs to be implemented in all child classes.

        Parameters
        ----------
        seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
            A seed to initialize the spectral noise. If None, then fresh, unpredictable
            entry will be pulled from the OS. If an int or array of ints, then it will
            be passed to a numpy.SeedSequency to derive the initial BitGenerator state.
            If a BitGenerator or Generator are passed, these will be wrapped and used
            instead. Default: None.

        Returns
        -------
        SpectralNoise
            A realization of the spectral noise, that can be shifted and evaluated.
        '''
        raise NotImplementedError()

class SpectralNoise(object):
    '''A spectral noise object.

    This object should not be used directly, but rather be made by a SpectralNoiseFactory object.
    '''
    def copy(self):
        '''Return a copy.

        Returns
        -------
        SpectralNoise
            A copy of ourselves.
        '''
        return copy.deepcopy(self)

    def shift(self, shift):
        '''In-place shift the noise along the grid axes.

        This function needs to be implemented by the child class.

        Parameters
        ----------
        shift : array_like
            The shift in the grid axes.
        '''
        raise NotImplementedError()

    def shifted(self, shift):
        '''Return a copy, shifted by `shift`.

        Parameters
        ----------
        shift : array_like
            The shift in the grid axes.

        Returns
        -------
        SpectralNoise
            A copy of ourselves, shifted by `shift`.
        '''
        a = self.copy()
        a.shift(shift)

        return a

    def __call__(self):
        '''Evaluate the noise on the pre-specified grid.

        This function should be implemented by all child classes.

        Returns
        -------
        Field
            The computed spectral noise.
        '''
        raise NotImplementedError()

class SpectralNoiseFactoryFFT(SpectralNoiseFactory):
    '''A spectral noise factory based on FFTs.

    The use of FFTs means that these spectral noises will wrap at the
    edge of the grid, i.e. they are continuous from one side of the grid
    to the other end. This means that the lowest frequencies of the PSD
    are not well represented and diminished in amplitude. The sampled
    spatial frequencies can be extended by increasing the oversample
    parameter at the cost of memory usage and computation time.

    See SpectralNoiseFactoryMultiscale for an alternative if this wrapping
    should be avoided at only a minor computational cost.

    Parameters
    ----------
    psd : Field generator
        The power spectral density of the noise.
    output_grid : Grid
        The grid on which to compute the noise.
    oversample : integer
        The amount by which to oversample to grid. For values higher than one,
        the spectral noise can be shifted by that fraction of the grid extent
        without wrapping.
    '''
    def __init__(self, psd, output_grid, oversample=1):
        SpectralNoiseFactory.__init__(self, psd, output_grid)

        if not self.output_grid.is_regular:
            raise ValueError("Can't make a SpectralNoiseFactoryFFT on a non-regular grid.")

        self.fourier = FastFourierTransform(self.output_grid, q=oversample)
        self.input_grid = self.fourier.output_grid

        self.period = output_grid.coords.delta * output_grid.coords.shape

        # * (2 * np.pi)**self.input_grid.ndim is due to conversion from PSD from "per Hertz" to "per radian", which yields a factor of 2pi per dimension
        self.C = np.sqrt(self.psd(self.input_grid) / self.input_grid.weights * (2 * np.pi)**self.input_grid.ndim)

    def make_random(self, seed=None):
        '''Make a single realization of the spectral noise.

        Parameters
        ----------
        seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
            A seed to initialize the spectral noise. If None, then fresh, unpredictable
            entry will be pulled from the OS. If an int or array of ints, then it will
            be passed to a numpy.SeedSequency to derive the initial BitGenerator state.
            If a BitGenerator or Generator are passed, these will be wrapped and used
            instead. Default: None.

        Returns
        -------
        SpectralNoiseFFT
            A realization of the spectral noise, that can be shifted and evaluated.
        '''
        rng = np.random.default_rng(seed)

        N = self.input_grid.size

        C = self.C * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        C = Field(C, self.input_grid)

        return SpectralNoiseFFT(self, C)

class SpectralNoiseFFT(SpectralNoise):
    '''A single realization of FFT spectral noise.

    Parameters
    ----------
    factory : SpectralNoiseFactoryMultiscale
        The factory used to generate this spectral noise instance.
    C : Field
        The PSD noise realization in Fourier space.
    '''
    def __init__(self, factory, C):
        self.factory = factory
        self.C = C

        self.coords = C.grid.separated_coords

    def shift(self, shift):
        '''In-place shift the noise along the grid axes.

        Parameters
        ----------
        shift : array_like
            The shift in the grid axes.
        '''
        S = [shift[i] * self.coords[i] for i in range(len(self.coords))]
        S = np.add.reduce(np.ix_(*S))

        self.C *= np.exp(-1j * S.ravel())

    def __call__(self):
        '''Evaluate the noise on the pre-specified grid.

        Returns
        -------
        Field
            The computed spectral noise.
        '''
        return self.factory.fourier.backward(self.C).real

class SpectralNoiseFactoryMultiscale(SpectralNoiseFactory):
    '''A spectral noise factory based on multiscale Fourier transforms.

    This factory should provide similar-looking noise compared to
    SpectralNoiseFactoryFFT, at a fraction of the computational cost.
    It does this by performing multiscale FTs rather than full-scale FTs.
    This means that the noise entropy is slightly reduced, however in the
    vast majority of cases completely invisibly so.

    Parameters
    ----------
    psd : Field generator
        The power spectral density of the noise.
    output_grid : Grid
        The grid on which to compute the noise.
    oversampling : integer
        The amount by which to oversample to grid. For values higher than one,
        the spectral noise can be shifted by that fraction of the grid extent
        without wrapping.
    '''
    def __init__(self, psd, output_grid, oversampling):
        SpectralNoiseFactory.__init__(self, psd, output_grid)

        self.oversampling = oversampling

        self.fourier_1 = FastFourierTransform(self.output_grid)
        self.input_grid_1 = self.fourier_1.output_grid

        self.input_grid_2 = self.input_grid_1.scaled(1.0 / oversampling)
        self.fourier_2 = MatrixFourierTransform(self.output_grid, self.input_grid_2)

        boundary = np.abs(self.input_grid_2.x).max()
        mask_1 = self.input_grid_1.as_('polar').r < boundary
        mask_2 = self.input_grid_2.as_('polar').r >= boundary

        # * (2*np.pi)**self.input_grid.ndim is due to conversion from PSD from "per Hertz" to "per radian", which yields a factor of 2pi per dimension
        self.C_1 = np.sqrt(psd(self.input_grid_1) / self.input_grid_1.weights * (2 * np.pi)**self.input_grid_1.ndim)
        self.C_1[mask_1] = 0
        self.C_2 = np.sqrt(psd(self.input_grid_2) / self.input_grid_2.weights * (2 * np.pi)**self.input_grid_1.ndim)
        self.C_2[mask_2] = 0

    def make_random(self, seed=None):
        '''Make a single realization of the spectral noise.

        Parameters
        ----------
        seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
            A seed to initialize the spectral noise. If None, then fresh, unpredictable
            entry will be pulled from the OS. If an int or array of ints, then it will
            be passed to a numpy.SeedSequency to derive the initial BitGenerator state.
            If a BitGenerator or Generator are passed, these will be wrapped and used
            instead. Default: None.

        Returns
        -------
        SpectralNoiseMultiscale
            A realization of the spectral noise, that can be shifted and evaluated.
        '''
        rng = np.random.default_rng(seed)

        N_1 = self.input_grid_1.size
        N_2 = self.input_grid_2.size

        C_1 = self.C_1 * (rng.standard_normal(N_1) + 1j * rng.standard_normal(N_1))
        C_2 = self.C_2 * (rng.standard_normal(N_2) + 1j * rng.standard_normal(N_2))

        return SpectralNoiseMultiscale(self, C_1, C_2)

class SpectralNoiseMultiscale(SpectralNoise):
    '''A single realization of multiscale spectral noise.

    Parameters
    ----------
    factory : SpectralNoiseFactoryMultiscale
        The factory used to generate this spectral noise instance.
    C_1 : Field
        The high-frequency part of the PSD noise realization.
    C_2 : Field
        The low-frequency part of the PSD noise realization.
    '''
    def __init__(self, factory, C_1, C_2):
        self.factory = factory

        self.C_1 = C_1
        self.C_2 = C_2

        self.coords_1 = C_1.grid.separated_coords
        self.coords_2 = C_2.grid.separated_coords

    def shift(self, shift):
        '''In-place shift the noise along the grid axes.

        Parameters
        ----------
        shift : array_like
            The shift in the grid axes.
        '''
        S_1 = [shift[i] * self.coords_1[i] for i in range(len(self.coords_1))]
        S_1 = sum(np.ix_(*S_1))

        S_2 = [shift[i] * self.coords_2[i] for i in range(len(self.coords_2))]
        S_2 = sum(np.ix_(*S_2))

        self.C_1 *= np.exp(-1j * S_1.ravel())
        self.C_2 *= np.exp(-1j * S_2.ravel())

    def __call__(self):
        '''Evaluate the noise on the pre-specified grid.

        Returns
        -------
        Field
            The computed spectral noise.
        '''
        ps = self.factory.fourier_1.backward(self.C_1).real
        ps += self.factory.fourier_2.backward(self.C_2).real

        return ps
