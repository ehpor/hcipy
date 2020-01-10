import numpy as np
import matplotlib.pyplot as plt 

from hcipy import * 

N = 512

#Defining the grids. 
pupil_grid = make_pupil_grid(N)
focal_grid = make_focal_grid(pupil_grid, 8, 5)

#The propagator.
prop = FraunhoferPropagator(pupil_grid, focal_grid)

#Aperture function
aperture = circular_aperture(1)(pupil_grid)

#The wavefront in the pupil.
pupil_wavefront = Wavefront(aperture) 

#Now we propagate it to the focal plane.
focal_wavefront = prop(pupil_wavefront)

#Setting the amount of photons per unit time.
focal_wavefront.total_power = 1000

#---------------------------------------------------------------------------------
#First example with scalars for the noise sources. 
#---------------------------------------------------------------------------------

flatfield = 0.05 # = 5% flat field error 
darkcurrentrate = 2 # = dark current counts per second
readnoise = 100 # = rms counts per read out
photonnoise = True

#Creating our detector.
detector_example_1 = NoisyDetector(input_grid=focal_grid, include_photon_noise=photonnoise, flat_field=flatfield, dark_current_rate=darkcurrentrate, read_noise=readnoise)

integration_time = np.logspace(1,6,6)

for t in integration_time:

    #Science measurement.
    detector_example_1.integrate(focal_wavefront, t)
    measurement = detector_example_1.read_out()

    plt.figure()
    imshow_field(measurement)
    plt.colorbar()
    plt.title('Example 1, t = ' + str(t) + ' seconds')

plt.show()

#---------------------------------------------------------------------------------
#Second example with arrays for the noise sources. 
#---------------------------------------------------------------------------------

flatfield = 1 + Field(np.random.rand(focal_wavefront.power.size) * 0.05, focal_grid)
darkcurrentrate = Field(np.random.rand(focal_wavefront.power.size) * 2, focal_grid)
readnoise = Field(np.random.rand(focal_wavefront.power.size) * 100, focal_grid)
photonnoise = True

plt.figure()
imshow_field(flatfield)
plt.colorbar()
plt.title('Flat field map')

plt.figure()
imshow_field(darkcurrentrate)
plt.colorbar()
plt.title('Dark current map')

plt.figure()
imshow_field(readnoise)
plt.colorbar()
plt.title('RMS read-noise map')

#Creating our detector.
detector_example_2 = NoisyDetector(input_grid=focal_grid, include_photon_noise=photonnoise, flat_field=flatfield, dark_current_rate=darkcurrentrate, read_noise=readnoise)

integration_time = np.logspace(1,6,6)

for t in integration_time:

    #Science measurement.
    detector_example_2.integrate(focal_wavefront, t)
    measurement = detector_example_2.read_out()

    plt.figure()
    imshow_field(measurement)
    plt.colorbar()
    plt.title('Example 2, t = ' + str(t) + ' seconds')

plt.show()
