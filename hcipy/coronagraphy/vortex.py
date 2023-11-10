import numpy as np
from scipy.signal import windows

from .multi_scale import MultiScaleCoronagraph
from ..optics import LinearRetarder, Apodizer, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field, field_dot
from ..aperture import make_circular_aperture
from ..fourier import FastFourierTransform, MatrixFourierTransform, FourierFilter

class VortexCoronagraph(MultiScaleCoronagraph):
    def __init__(self, input_grid, charge, lyot_stop=None, q=1024, scaling_factor=4, window_size=32):
        complex_mask = lambda grid: Field(np.exp(1j * charge * grid.as_('polar').theta), grid) * (1 - make_circular_aperture(1e-9)(grid))
        super().__init__(input_grid, complex_mask, lyot_stop, q, scaling_factor, window_size)

class VectorVortexCoronagraph(AgnosticOpticalElement):
    '''A vector vortex coronagraph.

    This :class:`OpticalElement` simulations the propagation of light through
    a vector vortex in the focal plane. To resolve the singularity of this vortex
    phase plate, a multi-scale approach is made. Discretisation errors made at
    a certain level are corrected by the next level with finer sampling.

    Parameters
    ----------
    charge : integer
        The charge of the vortex.
    lyot_stop : Field or OpticalElement
        The Lyot stop for the coronagraph. If it's a Field, it is converted to an
        OpticalElement for convenience. If this is None (default), then no Lyot stop is used.
    phase_retardation : scalar or function
        The phase retardation of the vector vortex plate, potentially as a
        function of wavelength. Changes of the phase retardation as a function
        of spatial position is not yet supported.
    q : scalar
        The minimum number of pixels per lambda/D. The number of levels in the multi-scale
        Fourier transforms will be chosen to reach at least this number of samples. The required
        q for a high-accuracy vortex coronagraph depends on the charge of the vortex. For charge 2,
        this can be as low as 32, but for charge 8 you need ~1024. Lower values give higher performance
        as a smaller number of levels is needed, but increases the sampling errors near the singularity.
        Charges not divisible by four require a much lower q. The default (q=1024) is conservative in
        most cases.
    scaling_factor : scalar
        The fractional increase in spatial frequency sampling per level. Larger scaling factors
        require a smaller number of levels, but each level requires a slower Fourier transform.
        Factors of 2 or 4 usually perform the best.
    window_size : integer
        The size of the next level in the number of pixels on the current layer. Lowering this
        increases performance in exchange for accuracy. Values smaller than 4-8 are not recommended.
    '''
    def __init__(self, charge, lyot_stop=None, phase_retardation=np.pi, q=1024, scaling_factor=4, window_size=32):
        self.charge = charge

        if hasattr(lyot_stop, 'forward') or lyot_stop is None:
            self.lyot_stop = lyot_stop
        else:
            self.lyot_stop = Apodizer(lyot_stop)

        self.phase_retardation = phase_retardation

        self.q = q
        self.scaling_factor = scaling_factor
        self.window_size = window_size

        AgnosticOpticalElement.__init__(self)

    def make_instance(self, instance_data, input_grid, output_grid, wavelength):
        pupil_diameter = input_grid.shape * input_grid.delta

        levels = int(np.ceil(np.log(self.q / 2) / np.log(self.scaling_factor))) + 1
        qs = [2 * self.scaling_factor**i for i in range(levels)]
        num_airys = [input_grid.shape / 2]

        focal_grids = []
        instance_data.props = []
        instance_data.jones_matrices = []

        for i in range(1, levels):
            num_airys.append(num_airys[i - 1] * self.window_size / (2 * qs[i - 1] * num_airys[i - 1]))

        for i in range(levels):
            q = qs[i]
            num_airy = num_airys[i]

            focal_grid = make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=1, focal_length=1)

            fast_axis_orientation = Field(self.charge / 2 * focal_grid.as_('polar').theta, focal_grid)
            retardance = self.evaluate_parameter(self.phase_retardation, input_grid, output_grid, wavelength)

            focal_mask_raw = LinearRetarder(retardance, fast_axis_orientation)
            jones_matrix = focal_mask_raw.jones_matrix

            jones_matrix *= 1 - make_circular_aperture(1e-9)(focal_grid)

            if i != levels - 1:
                wx = windows.tukey(self.window_size, 1, False)
                wy = windows.tukey(self.window_size, 1, False)
                w = np.outer(wy, wx)

                w = np.pad(w, (focal_grid.shape - w.shape) // 2, 'constant').ravel()
                jones_matrix *= 1 - w

            for j in range(i):
                fft = FastFourierTransform(focal_grids[j])
                mft = MatrixFourierTransform(focal_grid, fft.output_grid)

                jones_matrix -= mft.backward(fft.forward(instance_data.jones_matrices[j]))

            if i == 0:
                prop = FourierFilter(input_grid, jones_matrix, q)
            else:
                prop = FraunhoferPropagator(input_grid, focal_grid)

            focal_grids.append(focal_grid)
            instance_data.jones_matrices.append(jones_matrix)
            instance_data.props.append(prop)

    def get_input_grid(self, output_grid, wavelength):
        '''Get the input grid for a specified output grid and wavelength.

        This optical element only supports propagation to the same plane as
        its input.

        Parameters
        ----------
        output_grid : Grid
            The output grid of the optical element.
        wavelength : scalar or None
            The wavelength of the outgoing light.

        Returns
        -------
        Grid
            The input grid corresponding to the output grid and wavelength combination.
        '''
        return output_grid

    def get_output_grid(self, input_grid, wavelength):
        '''Get the output grid for a specified input grid and wavelength.

        This optical element only supports propagation to the same plane as
        its input.

        Parameters
        ----------
        input_grid : Grid
            The input grid of the optical element.
        wavelength : scalar or None
            The wavelength of the incoming light.

        Returns
        -------
        Grid
            The output grid corresponding to the input grid and wavelength combination.
        '''
        return input_grid

    @make_agnostic_forward
    def forward(self, instance_data, wavefront):
        '''Propagate a wavefront through the vortex coronagraph.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. This wavefront is expected to be
            in the pupil plane.

        Returns
        -------
        Wavefront
            The Lyot plane wavefront.
        '''
        wavelength = wavefront.wavelength
        wavefront.wavelength = 1

        for i, (jones_matrix, prop) in enumerate(zip(instance_data.jones_matrices, instance_data.props)):
            if i == 0:
                if not wavefront.is_polarized:
                    wf = Wavefront(wavefront.electric_field, input_stokes_vector=(1, 0, 0, 0))
                else:
                    wf = wavefront

                lyot = Wavefront(prop.forward(wf.electric_field), input_stokes_vector=wf.input_stokes_vector)
            else:
                focal = prop(wavefront)

                if not focal.is_polarized:
                    focal = Wavefront(focal.electric_field, input_stokes_vector=(1, 0, 0, 0))

                focal.electric_field = field_dot(jones_matrix, focal.electric_field)
                if i == 0:
                    lyot = prop.backward(focal)
                else:
                    lyot.electric_field += prop.backward(focal).electric_field

        lyot.wavelength = wavelength
        wavefront.wavelength = wavelength

        if self.lyot_stop is not None:
            lyot = self.lyot_stop.forward(lyot)

        return lyot

    @make_agnostic_backward
    def backward(self, instance_data, wavefront):
        '''Propagate backwards through the vortex coronagraph.

        This essentially is a forward propagation through the same vortex
        coronagraph, but with the sign of its charge flipped.

        Parameters
        ----------
        wavefront : Wavefront
            The Lyot plane wavefront.

        Returns
        -------
        Wavefront
            The pupil-plane wavefront.
        '''
        if self.lyot_stop is not None:
            wavefront = self.lyot_stop.backward(wavefront)

        wavelength = wavefront.wavelength
        wavefront.wavelength = 1

        for i, (jones_matrix, prop) in enumerate(zip(instance_data.jones_matrices, instance_data.props)):
            if i == 0:
                if not wavefront.is_polarized:
                    wavefront = Wavefront(wavefront.electric_field, input_stokes_vector=(1, 0, 0, 0))
                pup = Wavefront(
                    prop.backward(wavefront.electric_field),
                    input_stokes_vector=wavefront.input_stokes_vector
                )
            else:
                focal = prop(wavefront)
                if not focal.is_polarized:
                    focal = Wavefront(focal.electric_field, input_stokes_vector=(1, 0, 0, 0))

                focal.electric_field = field_dot(jones_matrix.conj(), focal.electric_field)
                pup.electric_field += prop.backward(focal).electric_field

        pup.wavelength = wavelength
        wavefront.wavelength = wavelength

        return pup

def make_ravc_masks(central_obscuration, charge=2, pupil_diameter=1, lyot_undersize=0):
    '''Make field generators for the pupil and Lyot-stop masks for a
    ring apodized vortex coronagraph.

    The formulas were implemented according to [Mawet2013]_.

    .. [Mawet2013] Dimitri Mawet et al. 2013 "Ring-apodized vortex coronagraphs for obscured telescopes. I. Transmissive
        ring apodizers" The Astrophysical Journal Supplement Series 209.1 (2013): 7

    Parameters
    ----------
    central_obscuration : scalar
        The diameter of the central obscuration.
    charge : integer
        The charge of the vortex coronagraph used.
    pupil_diameter : scalar
        The diameter of the pupil.
    lyot_undersize : scalar
        The fraction of the pupil diameter to which to undersize the Lyot stop.

    Returns
    -------
    pupil_mask : Field generator
        The complex transmission of the pupil mask.
    lyot_mask : Field generator
        The complex transmission of the Lyot-stop mask.
    '''
    R0 = central_obscuration / pupil_diameter

    if charge == 2:
        t1 = 1 - 0.25 * (R0**2 + R0 * np.sqrt(R0**2 + 8))
        R1 = R0 / np.sqrt(1 - t1)

        pupil1 = make_circular_aperture(pupil_diameter)
        pupil2 = make_circular_aperture(pupil_diameter * R1)
        co = make_circular_aperture(central_obscuration)
        pupil_mask = lambda grid: (pupil1(grid) * t1 + pupil2(grid) * (1 - t1)) * (1 - co(grid))

        lyot1 = make_circular_aperture(pupil_diameter * R1 + pupil_diameter * lyot_undersize)
        lyot2 = make_circular_aperture(pupil_diameter * (1 - lyot_undersize))
        lyot_stop = lambda grid: lyot2(grid) - lyot1(grid)
    elif charge == 4:
        R1 = np.sqrt(np.sqrt(R0**2 * (R0**2 + 4)) - 2 * R0**2)
        R2 = np.sqrt(R1**2 + R0**2)
        t1 = 0
        t2 = (R1**2 - R0**2) / (R1**2 + R0**2)

        pupil1 = make_circular_aperture(pupil_diameter)
        pupil2 = make_circular_aperture(pupil_diameter * R1)
        pupil3 = make_circular_aperture(pupil_diameter * R2)
        co = make_circular_aperture(central_obscuration)

        pupil_mask = lambda grid: (pupil1(grid) * t2 + pupil3(grid) * (t1 - t2) + pupil2(grid) * (1 - t1)) * (1 - co(grid))

        lyot1 = make_circular_aperture(pupil_diameter * R2 + pupil_diameter * lyot_undersize)
        lyot2 = make_circular_aperture(pupil_diameter * (1 - lyot_undersize))
        lyot_stop = lambda grid: lyot2(grid) - lyot1(grid)
    else:
        raise NotImplementedError()

    return pupil_mask, lyot_stop

def get_ravc_planet_transmission(central_obscuration_ratio, charge=2):
    '''Get the planet transmission for a ring-apodized vortex coronagraph.

    The formulas were implemented according to [Mawet2013]_.

    .. [Mawet2013] Dimitri Mawet et al. 2013 "Ring-apodized vortex coronagraphs for obscured telescopes. I. Transmissive
        ring apodizers" The Astrophysical Journal Supplement Series 209.1 (2013): 7

    Parameters
    ----------
    central_obscuration_ratio : scalar
        The ratio of the central obscuration diameter and the pupil diameter.
    charge : integer
        The charge of the vortex coronagraph used.

    Returns
    -------
    scalar
        The intensity transmission for a sufficiently off-axis point source
        for the ring-apodized vortex coronagraph. Point sources close to the vortex
        singularity will be lower in intensity.
    '''
    R0 = central_obscuration_ratio

    if charge == 2:
        t1_opt = 1 - 0.25 * (R0**2 + R0 * np.sqrt(R0**2 + 8))
        R1_opt = R0 / np.sqrt(1 - t1_opt)

        return t1_opt**2 * (1 - R1_opt**2) / (1 - (R0**2))
    elif charge == 4:
        R1 = np.sqrt(np.sqrt(R0**2 * (R0**2 + 4)) - 2 * R0**2)
        R2 = np.sqrt(R1**2 + R0**2)
        t2 = (R1**2 - R0**2) / (R1**2 + R0**2)

        return t2**2 * (1 - R2**2) / (1 - R0**2)
    else:
        raise NotImplementedError()
