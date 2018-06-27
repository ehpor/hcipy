from hcipy import *
import matplotlib.pyplot as plt
import numpy as np

def check_energy_conservation(shift_input, scale, shift_output, q, fov, dims):
	print(shift_input, scale, shift_output, q, fov, dims)
	
	grid = make_pupil_grid(dims).shifted(shift_input).scaled(scale)
	f_in = Field(np.random.randn(grid.size), grid)
	#f_in = Field(np.exp(-30 * grid.as_('polar').r**2), grid)

	fft = FastFourierTransform(grid, q=q, fov=fov, shift=shift_output)
	mft = MatrixFourierTransform(grid, fft.output_grid)
	nft = NaiveFourierTransform(grid, fft.output_grid)

	fourier_transforms = [fft, mft, nft]

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

			if False:#pattern_match > 0.0001:
				plt.subplot(1,3,1)
				imshow_field(f_in.imag)
				plt.colorbar()
				plt.subplot(1,3,2)
				imshow_field(f_inter.imag)
				plt.colorbar()
				plt.subplot(1,3,3)
				imshow_field(f_out.imag)
				plt.colorbar()
				plt.title(ft1.__class__.__name__ + ', ' + ft2.__class__.__name__)
				plt.show()

			
			energy_ratios.append(energy_ratio)
			patterns_match.append(pattern_match)

	print(np.array(energy_ratios).reshape((len(fourier_transforms), len(fourier_transforms))))
	print(np.array(patterns_match).reshape((len(fourier_transforms), len(fourier_transforms))))

if __name__ == '__main__':
	for shift_input in [[0,0],[0.1]]:
		for scale in [1,2]:
			for shift_output in [[0,0], [0.1]]:
				for q in [1,4]:
					for fov in [1,0.5,0.8]:
						for dims in [[8,8],[8,16]]:
							check_energy_conservation(shift_input, scale, shift_output, q, fov, dims)