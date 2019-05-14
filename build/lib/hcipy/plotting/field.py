from copy import copy
import numpy as np
from ..field import Field

def imshow_field(field, grid=None, ax=None, vmin=None, vmax=None, aspect='equal', norm=None, interpolation=None, non_linear_axes=False, cmap=None, mask=None, mask_color='k', *args, **kwargs):
	'''Display a two-dimensional image on a matplotlib figure.

	This function serves as an easy replacement for the matplotlib.pyplot.imshow() function.
	Its signature mostly folows that of matplotlib, with a few minor differences.

	Parameters
	----------
	field : Field or ndarray
		The field that we want to display. If this is an ndarray,
		then the parameter `grid` needs to be supplied. If the field
		is complex, then it will be automatically fed into :func:`complex_field_to_rgb`.
		If the field is a vector field with length 3 or 4, these will be 
		interpreted as an RGB or RGBA field.
	grid : Grid or None
		If a grid is supplied, it will be used instead of the grid of `field`.
	ax : matplotlib axes
		The axes which to draw on. If it is not given, the current axes will be used.
	vmin : scalar
		The minimum value on the colorbar. If it is not given, then the minimum value 
		of the field will be used.
	vmax : scalar
		The maximum value on the colorbar. If it is not given, then the maximum value 
		of the field will be used.
	aspect : ['auto', 'equal', scalar]
		If 'auto', changes the image aspect ratio to match that of the axes.
		If 'equal', changes the axes aspect ratio to match that of the image.
	norm : Normalize
		A Normalize instance is used to scale the input to the (0, 1) range for
		input to the `cmap`. If it is not given, a linear scale will be used.
	interpolation : string
		The interpolation method used. The default is 'nearest'. Supported values
		are {'nearest', 'bilinear'}.
	non_linear_axes : boolean
		If axes are scaled in a non-linear way, for example on a log plot, then imshow_field
		needs to use a more expensive implementation. This parameter is to indicate that this
		algorithm needs to be used.
	cmap : Colormap
		The colormap with which to plot the image. It is ignored if a complex
		field or a vector field is supplied.
	mask : field or ndarray
		If part of the image needs to be masked, this mask is overlayed on top of the image.
		This is for example useful when plotting a phase pattern on a certain aperture, which
		has no meaning outside of the aperture. Masks can be partially translucent, and will
		be automatically scaled between (0, 1). Zero means invisible, one means visible.
	mask_color : Color
		The color of the mask, if it is used.
	
	Returns
	-------
	AxesImage
		The produced image.
	'''
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from matplotlib.image import NonUniformImage

	if ax is None:
		ax = plt.gca()
	
	ax.set_aspect(aspect)

	if grid is None:
		grid = field.grid
	else:
		field = Field(field, grid)

	# If field is complex, draw complex
	if np.iscomplexobj(field):
		f = complex_field_to_rgb(field, rmin=vmin, rmax=vmax, norm=norm)
		vmin = None
		vmax = None
		norm = None
	else:
		f = field
	
	# Automatically determine vmin, vmax, norm if not overridden
	if norm is None and not np.iscomplexobj(field):
		if vmin is None:
			vmin = np.nanmin(f)
		if vmax is None:
			vmax = np.nanmax(f)
		norm = mpl.colors.Normalize(vmin, vmax)
	
	# Get extent
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separated and grid.is_('cartesian'):
		# We can draw this directly
		x, y = grid.coords.separated_coords
		z = f.shaped
		if np.iscomplexobj(field) or field.tensor_order > 0:
			z = np.rollaxis(z, 0, z.ndim)
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
		
		im = ax.pcolormesh(X, Y, z, *args, norm=norm, rasterized=True, cmap=cmap, **kwargs)
	else:
		# Use NonUniformImage to display
		im = NonUniformImage(ax, extent=(min_x, max_x, min_y, max_y), interpolation=interpolation , norm=norm, cmap=cmap, *args, **kwargs)
		im.set_data(x, y, z)

		from matplotlib.patches import Rectangle
		patch = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, facecolor='none')
		ax.add_patch(patch)
		im.set_clip_path(patch)

		ax.images.append(im)
	
	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)

	if mask is not None:
		one = np.ones(grid.size)
		col = mpl.colors.to_rgb(mask_color)

		m = np.array([one * col[0], one * col[1], one * col[2], 1 - mask / np.nanmax(mask)])

		imshow_field(m, grid, ax=ax)

	num_rows, num_cols = field.grid.shape
	def format_coord(x, y):
		col = int(np.round((x - min_x) / (max_x - min_x) * (num_cols - 1)))
		row = int(np.round((y - min_y) / (max_y - min_y) * (num_rows - 1)))

		if col >= 0 and col < num_cols and row >= 0 and row < num_rows:
			z = field.shaped[row, col]
			if np.iscomplexobj(z):
				return 'x=%0.3g, y=%0.3g, z=%0.3g + 1j * %0.3g = %0.3g * exp(1j * %0.2f)' % (x, y, z.real, z.imag, np.abs(z), np.angle(z))
			else:
				return 'x=%0.3g, y=%0.3g, z=%0.3g' % (x, y, z)
		return 'x=%0.3g, y=%0.3g' % (x, y)

	ax.format_coord = format_coord

	ax._sci(im)

	return im

def imsave_field(filename, field, grid=None, vmin=None, vmax=None, norm=None, mask=None, mask_color='w', cmap=None):
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	if grid is None:
		grid = field.grid
	else:
		field = Field(field, grid)
	
	# If field is complex, draw complex
	if np.iscomplexobj(field):
		f = complex_field_to_rgb(field, rmin=vmin, rmax=vmax, norm=norm)
		vmin = None
		vmax = None
		norm = None
	else:
		if norm is None:
			if vmin is None:
				vmin = np.nanmin(field)
			if vmax is None:
				vmax = np.nanmax(field)
			norm = mpl.colors.Normalize(vmin, vmax)
		f = field
	
	if mask is not None:
		f[~mask.astype('bool')] = np.nan

	cmap = copy(mpl.cm.get_cmap(cmap))
	cmap.set_bad(mask_color)

	plt.imsave(filename, f.shaped, cmap=cmap, vmin=vmin, vmax=vmax)

def contour_field(field, grid=None, ax=None, *args, **kwargs):
	import matplotlib.pyplot as plt
	if ax is None:
		ax = plt.gca()

	if grid is None:
		grid = field.grid
	else:
		field = Field(field, grid)
	
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separated and grid.is_('cartesian'):
		# We can contour directly
		x, y = grid.coords.separated_coords
		z = field.shaped
		
		X, Y = np.meshgrid(x, y)
		
		cs = ax.contour(X, Y, z, *args, **kwargs)
	else:
		raise NotImplementedError()

	return cs

def contourf_field(field, grid=None, ax=None, *args, **kwargs):
	import matplotlib.pyplot as plt
	if ax is None:
		ax = plt.gca()

	if grid is None:
		grid = field.grid
	else:
		field = Field(field, grid)
	
	c_grid = grid.as_('cartesian')
	min_x, min_y, max_x, max_y = c_grid.x.min(), c_grid.y.min(), c_grid.x.max(), c_grid.y.max()

	if grid.is_separated and grid.is_('cartesian'):
		# We can contour directly
		x, y = grid.coords.separated_coords
		z = field.shaped
		
		X, Y = np.meshgrid(x, y)
		
		cs = ax.contourf(X, Y, z, *args, **kwargs)
	else:
		raise NotImplementedError()

	return cs

def complex_field_to_rgb(field, theme='dark', rmin=None, rmax=None, norm=None):
	"""
	Takes an array of complex number and converts it to an array of [r, g, b],
	where phase gives hue and saturaton/value are given by the absolute value.
	Especially for use with imshow for complex plots.
	"""
	import matplotlib as mpl

	if not field.is_scalar_field:
		raise ValueError('Field must be a scalar field.')

	if norm is None:
		if rmin is None:
			rmin = np.nanmin(np.abs(field))
		if rmax is None:
			rmax = np.nanmax(np.abs(field))
		norm = mpl.colors.Normalize(rmin, rmax, True)
	
	hsv = np.zeros((field.size, 3), dtype='float')
	hsv[..., 0] = np.angle(field) / (2 * np.pi) % 1

	t = norm(np.abs(field))
	if theme == 'light':
		hsv[..., 1] = t
		hsv[..., 2] = 1
	elif theme == 'dark':
		hsv[..., 1] = 1
		hsv[..., 2] = t

	rgb = mpl.colors.hsv_to_rgb(hsv)
	alpha = np.isfinite(field)[:,np.newaxis]

	res = np.concatenate((rgb, alpha), axis=1)

	return Field(res.T, field.grid)