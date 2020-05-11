from hcipy import *
import numpy as np

def check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims):
	grid = make_uniform_grid(dims, 1).shifted(shift_input).scaled(scale)
	f_in = Field(np.random.randn(grid.size), grid).astype(dtype)

	energy_in = np.sum(np.abs(f_in)**2 * f_in.grid.weights)

	fft = FastFourierTransform(grid, q=q, fov=fov, shift=shift_output)
	mft = MatrixFourierTransform(grid, fft.output_grid)
	nft = NaiveFourierTransform(grid, fft.output_grid, True)
	nft2 = NaiveFourierTransform(grid, fft.output_grid, False)

	fourier_transforms = [fft, mft, nft, nft2]

	energy_ratios = []
	patterns_match = []
	for ft1 in fourier_transforms:
		f_inter = ft1.forward(f_in)
		assert f_inter.dtype == np.dtype(dtype)

		for ft2 in fourier_transforms:
			f_out = ft2.backward(f_inter)
			assert f_out.dtype == np.dtype(dtype)

			energy_out = np.sum(np.abs(f_out)**2 * f_out.grid.weights)
			energy_ratio = energy_out / energy_in

			pattern_match = np.std(np.abs(f_out - f_in)) / np.abs(f_in).mean()

			if fov == 1:
				# If the full fov is retained, energy and pattern should be conserved
				# for all fourier transform combinations.
				# Set different accuracy constraints depending on bit depth.

				print(dtype, shift_input, scale, shift_output, q, fov, dims)
				print(ft1.__class__.__name__, ft2.__class__.__name__)

				if np.dtype(dtype) == np.dtype('complex128'):
					assert pattern_match < 1e-13
					assert abs(energy_ratio - 1) < 1e-14
				else:
					assert pattern_match < 1e-6
					assert abs(energy_ratio - 1) < 1e-6

			energy_ratios.append(energy_ratio)
			patterns_match.append(pattern_match)

	energy_ratios = np.array(energy_ratios).reshape((len(fourier_transforms), len(fourier_transforms)))
	patterns_match = np.array(patterns_match).reshape((len(fourier_transforms), len(fourier_transforms)))

	print(energy_ratios - 1)
	print(patterns_match)

	# If the full fov is not retained, the pattern and energy loss should be the same
	# for all fourier transform combinations.
	if fov != 1:
		assert np.allclose(energy_ratios, energy_ratios[0, 0])
		assert np.allclose(patterns_match, patterns_match[0, 0])

def test_fourier_energy_conservation_1d():
	np.random.seed(0)

	for dtype in ['complex128', 'complex64']:
		for shift_input in [0,0.1]:
			for scale in [1,2]:
				for shift_output in [0,0.1]:
					for q in [1,3,4]:
						for fov in [1, 0.5, 0.8]:
							for dims in [64, 65]:
								check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims)

def test_fourier_energy_conservation_2d():
	np.random.seed(0)

	for dtype in ['complex128', 'complex64']:
		for shift_input in [[0,0],[0.1]]:
			for scale in [1,2]:
				for shift_output in [[0,0], [0.1]]:
					for q in [1,3,4]:
						for fov in [1,0.5,0.8]:
							for dims in [[8,8],[8,16],[9,9],[9,18]]:
								check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims)

def check_symmetry(scale, q, fov, dims):
	pass

def test_fourier_symmetries_2d():
	for scale in [1,2]:
		for q in [1,3,4]:
			for fov in [1,0.5,0.8]:
				for dims in [[8,8],[8,16],[9,9],[9,18]]:
					check_symmetry(scale, q, fov, dims)

def test_make_fourier_transform():
	input_grid = make_pupil_grid(128)

	ft = make_fourier_transform(input_grid, q=1, fov=1, planner='estimate')
	assert type(ft) == FastFourierTransform

	ft = make_fourier_transform(input_grid, q=8, fov=0.3, planner='estimate')
	assert type(ft) == MatrixFourierTransform

	ft = make_fourier_transform(input_grid, q=1, fov=1, planner='measure')
	ft = make_fourier_transform(input_grid, q=8, fov=0.1, planner='measure')

	output_grid = CartesianGrid(UnstructuredCoords([np.random.randn(100), np.random.randn(100)]))
	ft = make_fourier_transform(input_grid, output_grid)
	assert type(ft) == NaiveFourierTransform
