from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def fiber_principle():
	x = np.linspace(-np.pi/1, np.pi/1, 400)

	sigma = 0.8715
	mode = np.exp(-0.5 * (x/sigma)**2)

	f1 = lambda x: np.cos(x)
	f2 = lambda x: np.sin(2*x)
	f3 = lambda x: 0.85 * (np.cos(2.3*x)-0.134)

	funcs = [f1, f2, f3]

	shifts = np.linspace(-0.15, 0.15, 1000)

	for i,f in enumerate(funcs):
		y = f(x)
		z = mode * y

		c = [np.sum(f(x-s) * mode) for s in shifts]
		c = np.abs(c)

		plt.subplot(3,3,i+1)
		#plt.plot(x,0*x,'k', lw=3)
		plt.fill_between(x,0,z * (z>0), edgecolor='k', facecolor=colors.blue)
		plt.fill_between(x,0,z * (z<0), edgecolor='k', facecolor=colors.red)
		#plt.ylim(-1.1,1.1)
		plt.xlim(x.min()*1.1, x.max()*1.1)
		plt.gca().set_xticks([])
		plt.gca().set_yticks([])
		#plt.axis('off')

		plt.subplot(3,3,i+4)
		#plt.plot(x,0*x,'k', lw=3)
		plt.fill_between(x,0,z**2, edgecolor='k', facecolor=colors.blue)
		#plt.ylim(0,1.05)
		plt.xlim(x.min()*1.1, x.max()*1.1)
		plt.gca().set_xticks([])
		plt.gca().set_yticks([])
		#plt.axis('off')

		plt.subplot(3,3,i+7)
		plt.plot(shifts, c/c.max(), c=colors.blue)
		plt.yscale('log')
		plt.xlim(-0.16, 0.16)
		plt.ylim(1e-3,1.5)
		plt.gca().set_xticks([])
		plt.gca().set_yticks([])
	plt.show()

if __name__ == '__main__':
	fiber_principle()