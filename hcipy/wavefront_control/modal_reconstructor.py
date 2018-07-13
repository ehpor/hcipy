import numpy as np
import matplotlib.pyplot as plt

from ..field import make_pupil_grid
from ..optics import DeformableMirror
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated
from .observer import Observer

class ModalReconstructor(Observer):
	def __init__(self, mode_basis):
		if not hasattr(mode_basis, '__iter__'):
			self.mode_basis = [mode_basis]
		else:
			self.mode_basis = mode_basis
		
		self.flr = None
	
	def estimate(self, wavefront, t, filter_number=0):
		return self.flr.dot(wavefront)
	
	def set_filter(self,value,t):
		self.flr = value
