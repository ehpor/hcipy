import numpy as np
import matplotlib as mpl

def errorfill(x, y, y_err_pos, y_err_neg=None, color=None, alpha_fill=0.25, ax=None, **kwargs):
	"""
	Plot a line with filled errorbars.
	"""
	import matplotlib.pyplot as plt

	if ax is None:
		ax = plt.gca()

	if y_err_neg is None:
		y_err_neg = y_err_pos

	y_min = np.array(y) - np.array(y_err_neg)
	y_max = np.array(y) + np.array(y_err_pos)
	if color is None:
		l = ax.plot(x, y, **kwargs)
		color = l[0].get_color()
	else:
		l = ax.plot(x, y, color=color, **kwargs)

	facecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_fill)
	edgecolor = (0,0,0,0)
	ax.fill_between(x, y_max, y_min, edgecolor=edgecolor, facecolor=facecolor)

	return l