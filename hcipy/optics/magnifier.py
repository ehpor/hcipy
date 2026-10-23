from .optical_element import OpticalElement

class Magnifier(OpticalElement):
    def __init__(self, magnification):
        '''An ideal magnifier.

        This class magnifies a wavefront in an energy conserving manner.

        Parameters
        ----------
        magnification : scalar or function of wavelength
            The magnification of the system.
        '''
        self.magnification = magnification

    def _get_magnification(self, wavelength, xp):
        mag = self.magnification(wavelength) if callable(self.magnification) else self.magnification
        return mag * xp.ones(2)

    def forward(self, wavefront):
        wf = wavefront.copy()

        xp = wavefront.electric_field.grid.xp
        magnification = self._get_magnification(wavefront.wavelength, xp)

        wf.electric_field.grid = wf.electric_field.grid.scaled(magnification)
        wf.electric_field /= xp.sqrt(xp.prod(magnification))

        return wf

    def backward(self, wavefront):
        wf = wavefront.copy()

        xp = wavefront.electric_field.grid.xp
        magnification = self._get_magnification(wavefront.wavelength, xp)

        wf.electric_field.grid = wf.electric_field.grid.scaled(1.0 / magnification)
        wf.electric_field *= xp.sqrt(xp.prod(magnification))

        return wf

    def get_output_spec(self, input_spec):
        magnification = self._get_magnification(input_spec.wavelength, input_spec.grid.xp)

        grid = input_spec.grid.scaled(magnification)
        return input_spec.replace(grid=grid)

    def get_input_spec(self, output_spec):
        magnification = self._get_magnification(output_spec.wavelength, output_spec.grid.xp)

        grid = output_spec.grid.scaled(1.0 / magnification)
        return output_spec.replace(grid=grid)
