from hcipy import *
import numpy as np

def check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims):
	grid = make_uniform_grid(dims, 1, has_center=True).shifted(shift_input).scaled(scale)
	f_in = Field(np.random.randn(grid.size), grid).astype(dtype)

	energy_in = np.sum(np.abs(f_in)**2 * f_in.grid.weights)

	fft1 = FastFourierTransform(grid, q=q, fov=fov, shift=shift_output, emulate_fftshifts=True)
	fft2 = FastFourierTransform(grid, q=q, fov=fov, shift=shift_output, emulate_fftshifts=False)
	mft1 = MatrixFourierTransform(grid, fft1.output_grid, precompute_matrices=True, allocate_intermediate=True)
	mft2 = MatrixFourierTransform(grid, fft1.output_grid, precompute_matrices=True, allocate_intermediate=False)
	mft3 = MatrixFourierTransform(grid, fft1.output_grid, precompute_matrices=False, allocate_intermediate=True)
	mft4 = MatrixFourierTransform(grid, fft1.output_grid, precompute_matrices=False, allocate_intermediate=False)
	nft1 = NaiveFourierTransform(grid, fft1.output_grid, precompute_matrices=True)
	nft2 = NaiveFourierTransform(grid, fft1.output_grid, precompute_matrices=False)

	fourier_transforms = [fft1, fft2, mft1, mft2, mft3, mft4, nft1, nft2]

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

			energy_ratios.append(energy_ratio)
			patterns_match.append(pattern_match)

	energy_ratios = np.array(energy_ratios).reshape((len(fourier_transforms), len(fourier_transforms)))
	patterns_match = np.array(patterns_match).reshape((len(fourier_transforms), len(fourier_transforms)))

	if fov == 1:
		# When the full fov is retained, the pattern should be the same and energy should
		# be conserved. We use different accuracy limits based on bit depth.
		#print(np.max(patterns_match))
		if np.dtype(dtype) == np.dtype('complex128'):
			assert np.all(patterns_match < 1e-13)
			assert np.all(np.abs(energy_ratios - 1) < 1e-14)
		else:
			assert np.all(patterns_match < 1e-6)
			assert np.all(np.abs(energy_ratios - 1) < 1e-6)
	else:
		# If the full fov is not retained, the pattern and energy loss should be the same
		# for all fourier transform combinations.
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

def test_mft_precomputations():
	input_grid = make_pupil_grid(128)
	output_grid = make_fft_grid(input_grid, 1, 0.25)

	for precompute_matrices in [True, False]:
		for allocate_intermediate in [True, False]:
			mft = MatrixFourierTransform(input_grid, output_grid,
				precompute_matrices=precompute_matrices, allocate_intermediate=allocate_intermediate)

			mft.forward(input_grid.zeros())
			mft.forward(input_grid.ones())

			assert (mft.M1 is not None) == precompute_matrices
			assert (mft.intermediate_array is not None) == allocate_intermediate
