import numpy as np

def plot_kde_density_1d(x, bw_method=None, n_steps=200, alpha_fill=0.25, rug=False, overshoot=0.1, ax=None, kernel_size=None, **kwargs):
	"""
	Plot a 1D kernel density estimate function of the data.
	"""
	import matplotlib.pyplot as plt
	from scipy.stats import gaussian_kde

	if ax is None:
		ax = plt.gca()

	kernel = scipy.stats.gaussian_kde(x, bw_method)

	if not kernel_size is None:
		kernel.inv_cov /= kernel_size

	x_min, x_max = np.min(x), np.max(x)
	x_range = x_max - x_min

	x_eval = np.linspace(x_min - overshoot * x_range, x_max + overshoot * x_range, n_steps)

	y_eval = kernel(x_eval)

	res = ax.plot(x_eval, y_eval, **kwargs)

	if alpha_fill != 0:
		ax.fill_between(x_eval, y_eval, color=res[0].get_color(), alpha=alpha_fill)

	if rug:
		plot_rug(x, color=res[0].get_color(), ax=ax)

	return res

def plot_kde_density_2d(x, y, bw_method=None, n_steps=200, ax=None, overshoot=0.1, kernel_size=None, **kwargs):
	"""
	Plot a 2D kernel density estimate function of the data.
	"""
	import matplotlib.pyplot as plt
	from scipy.stats import gaussian_kde

	if ax is None:
		ax = plt.gca()

	xy = np.vstack((x, y))
	kernel = gaussian_kde(xy, bw_method)

	if not kernel_size is None:
		kernel.inv_cov /= kernel_size

	x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
	x_range, y_range = x_max - x_min, y_max - y_min

	x_eval = np.linspace(x_min - overshoot * x_range, x_max + overshoot * x_range, n_steps)
	y_eval = np.linspace(y_min - overshoot * y_range, y_max + overshoot * y_range, n_steps)

	X, Y = np.meshgrid(x_eval, y_eval)
	z_eval = kernel(np.vstack((X.flatten(), Y.flatten()))).reshape(X.shape)

	return plt.imshow(np.rot90(z_eval), extent=[np.min(x), np.max(x), np.min(y), np.max(y)], **kwargs)

def plot_rug(x, height=0.05, axis='x', color='k', ax=None):
	"""
	Plot a rug density plot efficiently. The height parameter is the length of 
	the rug marks given as a fraction of the axis length perpendicular to the 
	axis the marks are attached to.
	"""
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	if ax is None:
		ax = plt.gca()

	if axis == 'x':
		y_min, y_max = plt.ylim()
		y_max = (y_max - y_min) * height + y_min
		segments = [[(x_p, y_min), (x_p, y_max)] for x_p in x]
	else:
		x_min, x_max = plt.xlim()
		x_min = x_max - (x_max - x_min) * height
		segments = [[(x_min, x_p), (x_max, x_p)] for x_p in x]

	lc = mpl.collections.LineCollection(segments, colors=[mpl.colors.colorConverter.to_rgba(color)]*len(x))
	ax.add_collection(lc)
	return lc

# Thanks to https://gist.github.com/syrte
def plot_density_scatter(x, y, xsacle=1, yscale=1, sort=False, **kwargs):
	import matplotlib.pyplot as plt
	from scipy.stats import gaussian_kde

	kwargs.setdefault('edgecolor', 'none')
	if xscale != 1:
		x = x / xscale
	if yscale != 1:
		y = y / yscale
  
	xy = np.vstack([x, y])
	z = gaussian_kde(xy)(xy)
	z = z / (xscale * yscale)
	
	if sort:
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]
	
	return plt.scatter(x, y, c=z, **kwargs)