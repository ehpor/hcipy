from hcipy import *
import numpy as np

def check_energy_conservation(shift_input, scale, shift_output, q, fov, dims):
	print(shift_input, scale, shift_output, q, fov, dims)
	
	grid = make_uniform_grid(dims, 1).shifted(shift_input).scaled(scale)
	f_in = Field(np.random.randn(grid.size), grid)
	#f_in = Field(np.exp(-30 * grid.as_('polar').r**2), grid)

	fft = FastFourierTransform(grid, q=q, fov=fov, shift=shift_output)
	mft = MatrixFourierTransform(grid, fft.output_grid)
	nft = NaiveFourierTransform(grid, fft.output_grid, True)
	nft2 = NaiveFourierTransform(grid, fft.output_grid, False)

	fourier_transforms = [fft, mft, nft, nft2]

	energy_ratios = []
	patterns_match = []
	for ft1 in fourier_transforms:
		for ft2 in fourier_transforms:
			f_inter = ft1.forward(f_in)
			f_out = ft2.backward(f_inter)

			energy_in = np.sum(np.abs(f_in)**2 * f_in.grid.weights)
			energy_out = np.sum(np.abs(f_out)**2 * f_out.grid.weights)
			energy_ratio = energy_out / energy_in

			pattern_match = np.abs(f_out - f_in).max() / f_in.max()

			if fov == 1:
				# If the full fov is retained, energy and pattern should be conserved
				# for all fourier transform combinations.
				assert np.allclose(f_in, f_out)
				assert np.allclose(energy_in, energy_out)
			
			energy_ratios.append(energy_ratio)
			patterns_match.append(pattern_match)

	energy_ratios = np.array(energy_ratios).reshape((len(fourier_transforms), len(fourier_transforms)))
	patterns_match = np.array(patterns_match).reshape((len(fourier_transforms), len(fourier_transforms)))

	# If the full fov is not retained, the pattern and energy loss should be the same
	# for all fourier transform combinations.
	if fov != 1:
		assert np.allclose(energy_ratios, energy_ratios[0, 0])
		assert np.allclose(patterns_match, patterns_match[0, 0])

def test_fourier_energy_conservation_1d():
	for shift_input in [0,0.1]:
		for scale in [1,2]:
			for shift_output in [0,0.1]:
				for q in [1,3,4]:
					for fov in [1, 0.5, 0.8]:
						for dims in [64, 65]:
							check_energy_conservation(shift_input, scale, shift_output, q, fov, dims)

def test_fourier_energy_conservation_2d():
	for shift_input in [[0,0],[0.1]]:
		for scale in [1,2]:
			for shift_output in [[0,0], [0.1]]:
				for q in [1,3,4]:
					for fov in [1,0.5,0.8]:
						for dims in [[8,8],[8,16],[9,9],[9,18]]:
							check_energy_conservation(shift_input, scale, shift_output, q, fov, dims)

def check_symmetry(scale, q, fov, dims):
	pass

def test_fourier_symmetries_2d():
	for scale in [1,2]:
		for q in [1,3,4]:
			for fov in [1,0.5,0.8]:
				for dims in [[8,8],[8,16],[9,9],[9,18]]:
					check_symmetry(scale, q, fov, dims)