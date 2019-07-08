import numpy as np
from ..optics import OpticalElement, Wavefront, make_polychromatic

class MonochromaticPropagator(OpticalElement):
	def __init__(self, wavelength):
		self.wavelength = wavelength
