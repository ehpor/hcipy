from hcipy import *
import numpy as np
import pytest

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

@pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
def test_fourier_energy_conservation_1d(dtype):
	np.random.seed(0)

	for shift_input in [0, 0.1]:
		for scale in [1, 2]:
			for shift_output in [0, 0.1]:
				for q in [1, 1.23, 3, 4]:
					for fov in [1, 0.5, 0.8]:
						for dims in [64, 65]:
							check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims)

@pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
def test_fourier_energy_conservation_2d(dtype):
	np.random.seed(0)

	for shift_input in [[0, 0], [0.1]]:
		for scale in [1, 2]:
			for shift_output in [[0, 0], [0.1]]:
				for q in [1, 1.23, 3, 4]:
					for fov in [1, 0.5, 0.8]:
						for dims in [[8, 8], [8, 16], [9, 9], [9, 18]]:
							check_energy_conservation(dtype, shift_input, scale, shift_output, q, fov, dims)

def check_symmetry(scale, q, fov, dims):
	pass

def test_fourier_symmetries_2d():
	for scale in [1, 2]:
		for q in [1, 3, 4]:
			for fov in [1, 0.5, 0.8]:
				for dims in [[8, 8], [8, 16], [9, 9], [9, 18]]:
					check_symmetry(scale, q, fov, dims)

def test_make_fourier_transform():
	input_grid = make_pupil_grid(128)

	ft = make_fourier_transform(input_grid, q=1, fov=1, shift=0.1, planner='estimate')
	assert type(ft) == FastFourierTransform
	assert ft.input_grid == input_grid
	assert ft.output_grid == make_fft_grid(input_grid, q=1, fov=1, shift=0.1)

	fft_grid = make_fft_grid(input_grid, q=1, fov=1, shift=0.1)
	ft = make_fourier_transform(input_grid, fft_grid, planner='estimate')
	assert type(ft) == FastFourierTransform
	assert ft.input_grid == input_grid
	assert ft.output_grid == fft_grid

	ft = make_fourier_transform(input_grid, q=8, fov=0.3, shift=0.1, planner='estimate')
	assert type(ft) == MatrixFourierTransform
	assert ft.input_grid == input_grid
	assert ft.output_grid == make_fft_grid(input_grid, q=8, fov=0.3, shift=0.1)

	fft_grid = make_fft_grid(input_grid, q=8, fov=0.3, shift=0.1)
	ft = make_fourier_transform(input_grid, fft_grid, planner='estimate')
	assert type(ft) == MatrixFourierTransform
	assert ft.input_grid == input_grid
	assert ft.output_grid == fft_grid

	ft = make_fourier_transform(input_grid, q=1, fov=1, shift=0.1, planner='measure')
	ft = make_fourier_transform(input_grid, q=8, fov=0.1, shift=0.1, planner='measure')

	output_grid = CartesianGrid(UnstructuredCoords([np.random.randn(100), np.random.randn(100)]))
	ft = make_fourier_transform(input_grid, output_grid)
	assert type(ft) == NaiveFourierTransform
	assert ft.input_grid == input_grid
	assert ft.output_grid == output_grid

def test_fft_grid_reconstruction():
	for shift_input in [[0, 0], [0.1]]:
			for scale in [1, 2]:
				for shift_output in [[0, 0], [0.1]]:
					for q in [1, 1.234, 3, 4]:
						for fov in [1, 0.5, 0.8, [0.3, 0.23]]:
							for dims in [[8, 8], [8, 16], [9, 9], [9, 18]]:
								input_grid = make_uniform_grid(dims, 1, has_center=True).shifted(shift_input).scaled(scale)

								fft_grid = make_fft_grid(input_grid, q, fov, shift_output)

								assert is_fft_grid(fft_grid, input_grid)

								q_recon, fov_recon, shift_output_recon = get_fft_parameters(fft_grid, input_grid)
								fft_grid_recon = make_fft_grid(input_grid, q_recon, fov_recon, shift_output_recon)

								print(q, fov, shift_output)
								print(q_recon, fov_recon, shift_output_recon)

								assert np.allclose(fft_grid.x, fft_grid_recon.x)
								assert np.allclose(fft_grid.y, fft_grid_recon.y)

	# Check raising behaviour.
	input_grid = make_uniform_grid([128, 128], 1)
	output_grid = CartesianGrid(SeparatedCoords(input_grid.separated_coords))

	with pytest.raises(ValueError):
		get_fft_parameters(output_grid, input_grid)
	with pytest.raises(ValueError):
		get_fft_parameters(input_grid, output_grid)

	assert not is_fft_grid(output_grid, input_grid)
	assert not is_fft_grid(input_grid, output_grid)

	output_grid = make_fft_grid(input_grid, 1, 0.5).scaled(1.001)

	with pytest.raises(ValueError):
		get_fft_parameters(output_grid, input_grid)

	assert not is_fft_grid(output_grid, input_grid)

def test_fft_exceptions():
	regular_grid = make_uniform_grid([128, 128], 1)
	irregular_cartesian_grid = CartesianGrid(UnstructuredCoords([np.random.randn(1024), np.random.randn(1024)]))
	irregular_polar_grid = irregular_cartesian_grid.as_('polar')

	with pytest.raises(ValueError):
		FastFourierTransform(regular_grid, 0.9, 0.5)

	with pytest.raises(ValueError):
		FastFourierTransform(regular_grid, 2, 1.1)

	with pytest.raises(ValueError):
		FastFourierTransform(regular_grid, 2, -0.1)

	with pytest.raises(ValueError):
		FastFourierTransform(irregular_cartesian_grid, 2, 0.5)

	with pytest.raises(ValueError):
		FastFourierTransform(irregular_polar_grid, 2, 10.5)

def test_mft_precomputations():
	input_grid = make_pupil_grid(128)
	output_grid = make_fft_grid(input_grid, 1, 0.25)

	for precompute_matrices in [True, False]:
		for allocate_intermediate in [True, False]:
			mft = MatrixFourierTransform(
				input_grid, output_grid,
				precompute_matrices=precompute_matrices, allocate_intermediate=allocate_intermediate
			)

			mft.forward(input_grid.zeros())
			mft.forward(input_grid.ones())

			assert (mft.M1 is not None) == precompute_matrices
			assert (mft.intermediate_array is not None) == allocate_intermediate

def test_fourier_filter():
	for n in [16, 17, [16, 17]]:
		for q in [1, 2, 3]:
			for tensor_shape in [(), (3,), (3, 3)]:
				for scale in [1, 2]:
					input_grid = make_pupil_grid(n, scale)

					fft = FastFourierTransform(input_grid, q)

					tf_shape = tensor_shape + (fft.output_grid.size,)
					transfer_function = np.random.randn(*tf_shape) + 1j * np.random.randn(*tf_shape)
					transfer_function = Field(transfer_function, fft.output_grid)

					fourier_filter = FourierFilter(input_grid, transfer_function, q)

					f_shape = tensor_shape + (input_grid.size,)
					if len(tensor_shape) == 2:
						f_shape = f_shape[1:]
					f_in = Field(np.random.randn(*f_shape) + 1j * np.random.randn(*f_shape), input_grid)

					f_out_ff = fourier_filter.forward(f_in)
					f_in_ff = fourier_filter.backward(f_out_ff)

					if len(tensor_shape) == 2:
						ft = fft.forward(f_in)
						ft = field_dot(transfer_function, ft)
						f_out_fft = fft.backward(ft)

						ft = fft.forward(f_out_fft)
						ft = field_dot(field_conjugate_transpose(transfer_function), ft)
						f_in_fft = fft.backward(ft)
					else:
						f_out_fft = fft.backward(fft.forward(f_in) * transfer_function)
						f_in_fft = fft.backward(fft.forward(f_out_fft) * transfer_function.conj())

					assert np.allclose(f_out_fft, f_out_ff)
					assert np.allclose(f_in_fft, f_in_ff)
