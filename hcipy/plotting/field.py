import numpy as np

def imshow_field(field, grid=None, ax=None, vmin=None, vmax=None, aspect='equal', non_linear_axes=False, *args, **kwargs):
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from ..field import Field

	if ax is None:
		ax = plt.gca()
	
	ax.set_aspect(aspect)

	if grid is None:
		grid = data.grid
	else:
		field = Field(data, grid)
	
	if norm is None:
		if vmin is None:
			vmin = np.nanmin(field)
		if vmax is None:
			vmax = np.nanmax(field)
		norm = mpl.colors.Normalize(vmin, vmax)
	
	# Get extent
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separated and grid.is_('cartesian'):
		# We can draw this directly
		x, y = grid.coords.separated_coords
		z = field.shaped
	else:
		# We can't draw this directly. 
		raise NotImplementedError()
	
	if non_linear_axes:
		# Use pcolormesh to display
		x_mid = (x[1:] + x[:-1]) / 2
		y_mid = (y[1:] + y[:-1]) / 2
		
		x2 = np.concatenate(([1.5 * x[0] - 0.5 * x[1]], x_mid, [1.5 * x[-1] - 0.5 * x[-2]]))
		y2 = np.concatenate(([1.5 * y[0] - 0.5 * y[1]], y_mid, [1.5 * y[-1] - 0.5 * y[-2]]))
		X, Y = np.meshgrid(x2, y2)
		
		im = ax.pcolormesh(X, Y, z, norm=norm, *args, **kwargs)
	else:
		# Use NonUniformImage to display
		im = NonUniformImage(ax, extent=(min_x, max_x, min_y, max_y), interpolation=interpolation, norm=norm, *args, **kwargs)
		im.set_data(x, y, z)
		ax.images.append(im)
	
	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)

	plt.sci(im)

	return im

def contour_field(field, grid=None, ax=None, *args, **kwargs):
	if ax is None:
		ax = plt.gca()

	if grid is None:
		grid = data.grid
		values = data
	else:
		values = data
		data = SampledFunction(data, grid)
	
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separable and grid.is_('cartesian'):
		# We can contour directly
		x, y = grid.coords.separated_coords
		z = values.reshape(grid.shape).T
		
		X, Y = np.meshgrid(x, y)
		
		cs = ax.contour(X, Y, z, *args, **kwargs)
	else:
		raise NotImplementedError()

	return cs

def contourf_field(field, grid=None, ax=None, *args, **kwargs):
	if ax is None:
		ax = plt.gca()

	if grid is None:
		grid = data.grid
		values = data
	else:
		values = data
		data = SampledFunction(data, grid)
	
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separable and grid.is_('cartesian'):
		# We can contour directly
		x, y = grid.coords.separated_coords
		z = values.reshape(grid.shape).T
		
		X, Y = np.meshgrid(x, y)
		
		cs = ax.contourf(X, Y, z, *args, **kwargs)
	else:
		raise NotImplementedError()

	return cs