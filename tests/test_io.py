import numpy as np
import os 
from hcipy import *

def test_write_mode_basis():
    # grid for the test mode basis
    pupil_grid = make_pupil_grid(128)

    # generating a test mode basis 
    test_mode_basis = make_zernike_basis(num_modes=20, D=1, grid=pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True)

    # turning it into an array so we can compare 
    test_mode_basis_array = np.array([test_mode_basis])
    test_mode_basis_array = np.reshape(test_mode_basis_array, [20,128,128])

    file_name = 'write_mode_basis_test.fits'

    # saving the mode basis 
    write_mode_basis(test_mode_basis, file_name)

    # loading it very simply 
    test_mode_basis_array_read = read_fits(file_name)

    # comparing the arrays
    assert np.isclose(test_mode_basis_array, test_mode_basis_array_read, rtol=1e-02, atol=1e-05).all() 

    os.remove(file_name)

def test_read_mode_basis():
    #-------------------------------
    # testing a square mode basis that we read without providing a grid
    #-------------------------------

    # grid for the test mode basis
    pupil_grid_1 = make_pupil_grid(128)

    # testing a square mode basis defined in the pupil plane 
    test_mode_basis_1 = make_zernike_basis(num_modes=20, D=1, grid=pupil_grid_1, starting_mode=1, ansi=False, radial_cutoff=True)

    file_name_1 = 'read_mode_basis_test_1.fits'

    # writing the mode basis 
    write_mode_basis(test_mode_basis_1, file_name_1)

    # and loading it again 
    test_mode_basis_1_read = read_mode_basis(file_name_1, grid=None)

    # checking if the modes are still the same 
    for mode, mode_read in zip(test_mode_basis_1, test_mode_basis_1_read):
        assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()     

    # checking if the grid is correct 
    assert np.isclose(pupil_grid_1.x, test_mode_basis_1_read[0].grid.x, rtol=1e-02, atol=1e-05).all()     
    assert np.isclose(pupil_grid_1.y, test_mode_basis_1_read[0].grid.y, rtol=1e-02, atol=1e-05).all()     

    os.remove(file_name_1)

    #-------------------------------
    # testing a square mode basis that we read with providing a grid
    #-------------------------------

    # grid for the test mode basis
    pupil_grid_2 = make_pupil_grid(128, 3)

    # testing a square mode basis defined in the pupil plane 
    test_mode_basis_2 = make_zernike_basis(num_modes=20, D=3, grid=pupil_grid_2, starting_mode=1, ansi=False, radial_cutoff=True)

    file_name_2 = 'read_mode_basis_test_2.fits'

    # writing the mode basis 
    write_mode_basis(test_mode_basis_2, file_name_2)

    # and loading it again 
    test_mode_basis_2_read = read_mode_basis(file_name_2, grid=pupil_grid_2)

    # checking if the modes are still the same 
    for mode, mode_read in zip(test_mode_basis_2, test_mode_basis_2_read):
        assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()     

    # checking if the grid is correct 
    assert np.isclose(pupil_grid_2.x, test_mode_basis_2_read[0].grid.x, rtol=1e-02, atol=1e-05).all()     
    assert np.isclose(pupil_grid_2.y, test_mode_basis_2_read[0].grid.y, rtol=1e-02, atol=1e-05).all()     

    os.remove(file_name_2)


    #-------------------------------
    # testing a non-square mode basis that we read with providing a grid
    #-------------------------------

    # grid for the test mode basis
    pupil_grid_3 = make_uniform_grid([128,256], [128,256], center=0, has_center=False)

    # testing a square mode basis defined in the pupil plane 
    test_mode_basis_3 = []

    for i in np.arange(20):
        test_mode_basis_3.append(Field(np.random.rand(128*256), pupil_grid_3))

    test_mode_basis_3 = ModeBasis(test_mode_basis_3)

    file_name_3 = 'read_mode_basis_test_3.fits'

    # writing the mode basis 
    write_mode_basis(test_mode_basis_3, file_name_3)

    # and loading it again 
    test_mode_basis_3_read = read_mode_basis(file_name_3, grid=pupil_grid_3)

    # checking if the modes are still the same 
    for mode, mode_read in zip(test_mode_basis_3, test_mode_basis_3_read):
        assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()     

    # checking if the grid is correct 
    assert np.isclose(pupil_grid_3.x, test_mode_basis_3_read[0].grid.x, rtol=1e-02, atol=1e-05).all()     
    assert np.isclose(pupil_grid_3.y, test_mode_basis_3_read[0].grid.y, rtol=1e-02, atol=1e-05).all()     

    os.remove(file_name_3)
