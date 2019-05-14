import numpy as np
from ..optics import OpticalElement
from ..mode_basis import ModeBasis
from ..math_util import inverse_truncated

class PerfectCoronagraph(OpticalElement):
	'''A perfect coronagraph for a certain aperture and order.

		These type of coronagraphs suppress all light for a flat wavefront. The incoming complex
		amplitude :math:`A` is modified as follows (following [1]_):

		.. math::
			\overline{A} = A - \Pi \sqrt{S}
		
		where :math:`\overline{A}` is the resulting complex ampliutude, :math:`\Pi` is the telescope
		pupil, and :math:`S` is the Strehl ratio of the incoming wavefront.

		Higher orders are added by fitting higher-order electric field modes to the incoming
		wavefront and subtracting those, following [2]_.

		.. [1] Celine Cavarroc et al. "Fundamental limitations on Earth-like planet detection with extremely large telescopes." Astronomy & Astrophysics 447.1 (2006): 397-403
		.. [2] Olivier Guyon et al. "Theoretical limits on extrasolar terrestrial planet detection with coronagraphs." The Astrophysical Journal Supplement Series 167.1 (2006): 81

		Parameters
		----------
		aperture : Field
			The reference aperture. The perfect coronagraph is designed for this aperture.
		order : integer
			The order of the perfect coronagraph. This must be even.
		coeffs : list or ndarray or None
			The coefficients that are used for subtraction. This allows for partial suppression of certain
			modes, which can be used to design perfect coronagraphs that are insensitive to stellar
			radius [2]_. If it is None, all modes are completely suppressed.
		'''
	def __init__(self, aperture, order=2, coeffs=None):
		self.pupil_grid = aperture.grid
		modes = []

		if coeffs is not None:
			order = int(2 * np.ceil(0.5 * (np.sqrt(8*len(coeffs) + 1) - 1)))
			self.coeffs = coeffs
		else:
			self.coeffs = np.ones(int(order * (order / 2 + 1) / 4))

		for i in range(order // 2):
			for j in range(i + 1):
				modes.append(aperture * self.pupil_grid.x**j * self.pupil_grid.y**(i-j))
		
		self.mode_basis = ModeBasis(modes).orthogonalized

		self.transformation = self.mode_basis.transformation_matrix
		self.transformation_inverse = inverse_truncated(self.transformation, 1e-6)

	def forward(self, wavefront):
		'''Propagate the wavefront through the perfect coronagraph.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront (in the pupil plane)
		
		Returns
		-------
		Wavefront
			The post-coronagraphic wavefront (in the pupil plane).
		'''
		wf = wavefront.copy()

		correction = self.transformation.dot(self.coeffs * self.transformation_inverse.dot(wf.electric_field))
		wf.electric_field -= correction

		return wf
	
	def backward(self, wavefront):
		'''Propagate the wavefront backwards through the perfect coronagraph.

		This method behaves the same as the forward propagation.

		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront (in the pupil plane)
		
		Returns
		-------
		Wavefront
			The post-coronagraphic wavefront (in the pupil plane).
		'''
		return self.forward(wavefront)
	
	def get_transformation_matrix_forward(self, wavelength=1):
		'''Get the forward propagation transformation matrix.

		Parameters
		----------
		wavelength : scalar
			The wavelength at which to calculate the transformation matrix.
		
		Returns
		-------
		ndarray
			The forward transformation_matrix.
		'''
		return np.eye(self.pupil_grid.size) - self.transformation.dot(self.coeffs * self.transformation_inverse)
	
	def get_transformation_matrix_backward(self, wavelength=1):
		'''Get the backwards propagation transformation matrix.

		This method behaves the same as the forward propagation.

		Parameters
		----------
		wavelength : scalar
			The wavelength at which to calculate the transformation matrix.
		
		Returns
		-------
		ndarray
			The backward transformation_matrix.
		'''
		return self.get_transformation_matrix_forward(wavelength)