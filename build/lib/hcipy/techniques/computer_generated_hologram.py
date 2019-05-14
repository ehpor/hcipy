from ..field import Field

import numpy as np

def make_cgh(pupil_grid, mode, position, amplitude):
    O = 2*np.pi * np.dot(position, pupil_grid.coords)
    R = amplitude * mode

    hologram = np.sqrt(2) * np.cos(R + O)
    return Field(hologram, pupil_grid)

def make_multiplexed_cgh(pupil_grid, modes, positions, amplitudes):
    amplitudes = np.ones(len(modes)) * amplitudes

    cghs = [make_cgh(pupil_grid, mode, position, amplitude) for mode, position, amplitude in zip(modes, positions, amplitudes)]

    return Field(np.sum(cghs, axis=0), pupil_grid)