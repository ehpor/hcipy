import numpy as np

from ..optics import OpticalElement, Wavefront, LinearRetarder, JonesMatrixOpticalElement, AgnosticOpticalElement, make_agnostic_forward, make_agnostic_backward
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field, make_pupil_grid
from ..aperture import circular_aperture
from ..fourier import FastFourierTransform, MatrixFourierTransform

class VortexCoronagraph(OpticalElement):
	'''An optical vortex coronagraph.

	This :class:`OpticalElement` simulations the propagation of light through
	a vortex in the focal plane. To resolve the singularity of this vortex
	phase plate, a multi-scale approach is made. Discretisation errors made at
	a certain level are corrected by the next level with finer sampling.

	Parameters
	----------
	input_grid : Grid
		The grid on which the incoming wavefront is defined.
	charge : integer
		The charge of the vortex.
	levels : integer
		The number of levels in the multi-scale sampling.
	scaling_factor : scalar
		The fractional increase in spatial frequency sampling per level.
	'''
	def __init__(self, input_grid, charge=2, levels=4, scaling_factor=4):
		self.input_grid = input_grid
		
		q_outer = 2
		num_airy_outer = input_grid.shape[0] / 2
		pupil_diameter = input_grid.shape * input_grid.delta

		focal_grids = []
		self.focal_masks = []
		self.props = []

		for i in range(levels):
			q = q_outer * scaling_factor**i
			if i == 0:
				num_airy = num_airy_outer
			else:
				num_airy = 32.0 / (q_outer * scaling_factor**(i-1))

			focal_grid = make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=1, focal_length=1)
			focal_mask = Field(np.exp(1j * charge * focal_grid.as_('polar').theta), focal_grid)

			focal_mask *= 1 - circular_aperture(1e-9)(focal_grid)

			for j in range(i):
				fft = FastFourierTransform(focal_grids[j])
				mft = MatrixFourierTransform(focal_grid, fft.output_grid)

				focal_mask -= mft.backward(fft.forward(self.focal_masks[j]))
			
			prop = FraunhoferPropagator(input_grid, focal_grid)

			focal_grids.append(focal_grid)
			self.focal_masks.append(focal_mask)
			self.props.append(prop)

	def forward(self, wavefront):
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

		for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
			focal = prop(wavefront)
			focal.electric_field *= mask
			if i == 0:
				lyot = prop.backward(focal)
			else:
				lyot.electric_field += prop.backward(focal).electric_field

		lyot.wavelength = wavelength
		return lyot
	
	def backward(self, wavefront):
		'''Propagate backwards through the vortex coronagraph.

		This essentially is a forward propagation through a the same vortex
		coronagraph, but with the sign of the its charge flipped.

		Parameters
		----------
		wavefront : Wavefront
			The Lyot plane wavefront.
		
		Returns
		-------
		Wavefront
			The pupil-plane wavefront.
		'''
		wavelength = wavefront.wavelength
		wavefront.wavelength = 1

		for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
			focal = prop(wavefront)
			focal.electric_field *= mask.conj()
			if i == 0:
				pup = prop.backward(focal)
			else:
				pup.electric_field += prop.backward(focal).electric_field

		pup.wavelength = wavelength
		return pup

class VectorVortexCoronagraph(AgnosticOpticalElement):
	'''An vector vortex coronagraph.

	This :class:`OpticalElement` simulations the propagation of light through
	a vector vortex in the focal plane. To resolve the singularity of this vortex
	phase plate, a multi-scale approach is made. Discretisation errors made at
	a certain level are corrected by the next level with finer sampling.

	Parameters
	----------
	charge : integer
		The charge of the vortex.
	phase_retardation : scalar or function
		The phase retardation of the vector vortex plate, potentially as a
		function of wavelength. Changes of the phase retardation as a function
		of spatial position is not yet supported.
	levels : integer
		The number of levels in the multi-scale sampling.
	scaling_factor : scalar
		The fractional increase in spatial frequency sampling per level.
	'''
	def __init__(self, charge=2, phase_retardation=np.pi, levels=4, scaling_factor=4):
		self.charge = charge
		self.phase_retardation = phase_retardation
		self.levels = levels
		self.scaling_factor = scaling_factor

		AgnosticOpticalElement.__init__(self)

	def make_instance(self, instance_data, input_grid, output_grid, wavelength):
		q_outer = 2
		num_airy_outer = input_grid.shape[0] / 2
		pupil_diameter = input_grid.shape * input_grid.delta

		focal_grids = []
		jones_matrices = []
		instance_data.props = []
		instance_data.focal_masks = []

		for i in range(self.levels):
			q = q_outer * self.scaling_factor**i

			if i == 0:
				num_airy = num_airy_outer
			else:
				num_airy = 32.0 / (q_outer * self.scaling_factor**(i - 1))

			focal_grid = make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=1, focal_length=1)

			fast_axis_orientation = Field(self.charge / 2 * focal_grid.as_('polar').theta, focal_grid)
			retardance = self.evaluate_parameter(self.phase_retardation, input_grid, output_grid, wavelength)

			focal_mask_raw = LinearRetarder(retardance, fast_axis_orientation)
			jones_matrix = focal_mask_raw.jones_matrix
			jones_matrix *= 1 - circular_aperture(1e-9)(focal_grid)

			def eval_jones(jones_matrix, jones_matrices, focal_grids):
				mat = jones_matrix.copy()

				for j in range(i):
					fft = FastFourierTransform(focal_grids[j])
					mft = MatrixFourierTransform(focal_grid, fft.output_grid)

					mat -= mft.backward(fft.forward(jones_matrices[j]))

				return mat

			jones_matrices.append(focal_mask_raw.construct_function(eval_jones, jones_matrix, jones_matrices, focal_grids))
			focal_mask = JonesMatrixOpticalElement(jones_matrices[-1])

			prop = FraunhoferPropagator(input_grid, focal_grid)

			focal_grids.append(focal_grid)
			instance_data.focal_masks.append(focal_mask)
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

		for i, (mask, prop) in enumerate(zip(instance_data.focal_masks, instance_data.props)):
			focal = prop(wavefront)

			focal.wavelength = wavelength
			focal = mask.forward(focal)
			focal.wavelength = 1

			if i == 0:
				lyot = prop.backward(focal)
			else:
				lyot.electric_field += prop.backward(focal).electric_field

		lyot.wavelength = wavelength
		return lyot

	@make_agnostic_backward
	def backward(self, instance_data, wavefront):
		'''Propagate backwards through the vortex coronagraph.

		This essentially is a forward propagation through a the same vortex
		coronagraph, but with the sign of the its charge flipped.

		Parameters
		----------
		wavefront : Wavefront
			The Lyot plane wavefront.

		Returns
		-------
		Wavefront
			The pupil-plane wavefront.
		'''
		wavelength = wavefront.wavelength
		wavefront.wavelength = 1

		for i, (mask, prop) in enumerate(zip(instance_data.focal_masks, instance_data.props)):
			focal = prop(wavefront)

			focal.wavelength = wavelength
			focal = mask.backward(focal)
			focal.wavelength = 1

			if i == 0:
				pup = prop.backward(focal)
			else:
				pup.electric_field += prop.backward(focal).electric_field

			pup.wavelength = wavelength
			return pup

def make_ravc_masks(central_obscuration, charge=2, pupil_diameter=1, lyot_undersize=0):
	'''Make field generators for the pupil and Lyot-stop masks for a
	ring apodized vortex coronagraph.

	The formulas were implemented according to [1]_.
	
	.. [1] Dimitri Mawet et al. 2013 "Ring-apodized vortex coronagraphs for obscured telescopes. I. Transmissive ring apodizers" The Astrophysical Journal Supplement Series 209.1 (2013): 7

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

		pupil1 = circular_aperture(pupil_diameter)
		pupil2 = circular_aperture(pupil_diameter * R1)
		co = circular_aperture(central_obscuration)
		pupil_mask = lambda grid: (pupil1(grid) * t1 + pupil2(grid) * (1 - t1)) * (1 - co(grid))

		lyot1 = circular_aperture(pupil_diameter * R1 + pupil_diameter * lyot_undersize)
		lyot2 = circular_aperture(pupil_diameter * (1 - lyot_undersize))
		lyot_stop = lambda grid: lyot2(grid) - lyot1(grid)
	elif charge == 4:
		R1 = np.sqrt(np.sqrt(R0**2 * (R0**2 + 4)) - 2*R0**2)
		R2 = np.sqrt(R1**2 + R0**2)
		t1 = 0
		t2 = (R1**2 - R0**2) / (R1**2 + R0**2)

		pupil1 = circular_aperture(pupil_diameter)
		pupil2 = circular_aperture(pupil_diameter * R1)
		pupil3 = circular_aperture(pupil_diameter * R2)
		co = circular_aperture(central_obscuration)

		pupil_mask = lambda grid: (pupil1(grid) * t2 + pupil3(grid) * (t1 - t2) + pupil2(grid) * (1 - t1)) * (1 - co(grid))

		lyot1 = circular_aperture(pupil_diameter * R2 + pupil_diameter * lyot_undersize)
		lyot2 = circular_aperture(pupil_diameter * (1 - lyot_undersize))
		lyot_stop = lambda grid: lyot2(grid) - lyot1(grid)
	else:
		raise NotImplementedError()

	return pupil_mask, lyot_stop

def get_ravc_planet_transmission(central_obscuration_ratio, charge=2):
	'''Get the planet transmission for a ring-apodized vortex coronagraph.

	The formulas were implemented according to [1]_.
	
	.. [1] Dimitri Mawet et al. 2013 "Ring-apodized vortex coronagraphs for obscured telescopes. I. Transmissive ring apodizers" The Astrophysical Journal Supplement Series 209.1 (2013): 7

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
		R1 = np.sqrt(np.sqrt(R0**2 * (R0**2 + 4)) - 2*R0**2)
		R2 = np.sqrt(R1**2 + R0**2)
		t1 = 0
		t2 = (R1**2 - R0**2) / (R1**2 + R0**2)

		return t2**2 * (1 - R2**2) / (1 - R0**2)
	else:
		raise NotImplementedError()
