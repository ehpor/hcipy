from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

r_0 = 0.02
N = 256
F_mla = 0.03
N_mla = 8

D_mla = 1.0 / N_mla

pupil_grid = make_pupil_grid(N, 1.3)
focal_grid = make_focal_grid(pupil_grid, 8, 8)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

x = np.arange(-1,1,D_mla)
mla_grid = CartesianGrid(SeparatedCoords((x,x)))
mla_shape = rectangular_aperture(D_mla)
microlens_array = MicroLensArray(pupil_grid, mla_grid, F_mla * D_mla, mla_shape)
sh_prop = FresnelPropagator(pupil_grid, F_mla * D_mla)

aperture = circular_aperture(1)

spectral_noise_factory = SpectralNoiseFactoryFFT(kolmogorov_psd, pupil_grid, 8)
turbulence_layers = make_standard_multilayer_atmosphere(r_0, wavelength=1)
atmospheric_model = AtmosphericModel(spectral_noise_factory, turbulence_layers)

wf = Wavefront(aperture(pupil_grid))
variances = []

times = np.linspace(0,0.7,200)
mw = GifWriter('adaptive_optics2.gif', 10)
for t in times:
	atmospheric_model.t = t
	wf2 = atmospheric_model(wf)

	sci_img = prop(wf2).intensity
	wf3 = microlens_array(wf2)
	wfs_img = sh_prop(wf3).intensity

	#imshow_field(wf3.phase)
	#plt.colorbar()
	#plt.show()

	plt.clf()
	plt.subplot(2,2,1)
	imshow_field(np.log10(sci_img / sci_img.max()), vmin=-5)
	plt.subplot(2,2,2)
	imshow_field(wfs_img / wfs_img.max())
	plt.subplot(2,2,3)
	imshow_field(wf2.phase * aperture(pupil_grid), vmin=-np.pi, vmax=np.pi, cmap='RdBu')
	plt.draw()
	plt.pause(0.00001)
	#mw.add_frame()
mw.close()