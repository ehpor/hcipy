import numpy as np

class MonochromaticPropagator(object):
    def __init__(self, wavelength):
        self.wavelength = wavelength
    
    def __call__(self, wavefront):
        return self.forward(wavefront)
    
    def forward(self, wavefront):
        raise NotImplementedError()
    
    def backward(self, wavefront):
        raise NotImplementedError()

 def make_polychromatic_propagator(monochromatic_propagator):
    class PolychromaticPropagator(object):
        def __init__(self, *args, **kwargs):
            self.wavelengths = []
            self.monochromatic_propagators = []
            self.monochromatic_args = args
            self.monochromatic_kwargs = kwargs
        
        def get_monochromatic_propagator(self, wavelength):
            if len(self.wavelengths) > 0:
                i = np.argmin(np.abs(wavelength - np.array(wavelengths))
                return self.monochromatic_propagators[i]
            
            m = monochromatic_propagator(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)

            self.wavelengths.append(wavelength)
            self.monochromatic_propagators(m)

            return m

        def __call__(self, wavefront):
            return self.forward(wavefront)
        
        def forward(self, wavefront):
            return self.get_monochromatic_propagator(wavefront.wavelength).forward(wavefront)
        
        def backward(self, wavefront):
            return self.get_monochromatic_propagator(wavefront.wavelength).backward(wavefront)

    return PolychromaticPropagator