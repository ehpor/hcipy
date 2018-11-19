import numpy as np 
from hcipy import *

def test_statistics_noisy_detector():

    N = 512
    grid = make_pupil_grid(N)

    test_wavefront = Field(np.ones(N**2), grid)
    
    #First we test photon noise, dark current noise and read noise.
    flat_field = 0
    dark_currents = np.logspace(1,6,6)
    read_noises = np.logspace(1,6,6)
    photon_noise = True

    for dc in dark_currents:
        for rn in read_noises: 

            #The test detector.
            test_detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=flat_field, dark_current_rate=dc, read_noise=rn)

            #The integration times we will test.
            integration_time = np.logspace(1,6,6)

            for t in integration_time:
                # integration
                test_detector.integrate(test_wavefront, t)

                # read out 
                measurement = test_detector.read_out()

                # The std of the data by the detector.
                std_measurement = np.std(measurement)

                # The std that we expect given the input.
                expected_std = np.sqrt(test_wavefront[0] * t + rn**2 + dc * t)

                assert np.isclose(expected_std, std_measurement, rtol=1e-02, atol=1e-05)z

    #Now we test the flat field separately.
    
    flat_fields = np.linspace(0,1,100)
    dark_current = 0
    read_noise = 0
    photon_noise = False

    for ff in flat_fields:
        #The test detector.
        test_detector = NoisyDetector(input_grid=grid, include_photon_noise=photon_noise, flat_field=ff, dark_current_rate=dark_current, read_noise=read_noise)

        #The integration times we will test.
        integration_time = np.logspace(1,6,6)

        for t in integration_time:
            # integration
            test_detector.integrate(test_wavefront, t)

            # read out 
            measurement = test_detector.read_out()

            # The std of the data by the detector.
            std_measurement = np.std(measurement)

            # The std that we expect given the input.
            expected_std = ff * test_wavefront[0] * t 

            assert np.isclose(expected_std, std_measurement, rtol=1e-02, atol=1e-05)
 

 