from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

def check_energy_conservation(shift, scale):
	grid = make_pupil_grid(32).shifted(shift).scaled(scale)
	f_in = Field(np.random.randn(grid.size), grid)

	fft = FastFourierTransform(grid)
	mft = MatrixFourierTransform(grid, fft.output_grid)
	nft = NaiveFourierTransform(grid, fft.output_grid)

	fourier_transforms = [fft, mft, nft]

	energy_ratios = []
	patterns_match = []
	for ft1 in fourier_transforms:
		for ft2 in fourier_transforms:
			f_out = ft2.backward(ft1.forward(f_in))

			energy_in = np.sum(np.abs(f_in)**2 * f_in.grid.weights)
			energy_out = np.sum(np.abs(f_out)**2 * f_out.grid.weights)
			energy_ratio = energy_out / energy_in

			pattern_match = np.abs(f_out - f_in).max() / f_in.max()
			
			energy_ratios.append(energy_ratio)
			patterns_match.append(pattern_match)

	print(np.array(energy_ratios).reshape(3,3))
	print(np.array(patterns_match).reshape(3,3))

if __name__ == '__main__':
	check_energy_conservation([0,0],1)
	check_energy_conservation([0.1,0],1)
	check_energy_conservation([0,0],2)
