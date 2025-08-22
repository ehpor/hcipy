from __future__ import division
import functools
import inspect

import numpy as np
from matplotlib.path import Path

from ..field import Field, make_hexagonal_grid, CartesianGrid, PolarGrid, UnstructuredCoords, SeparatedCoords
from ..dev import deprecated_name_changed

def make_circular_aperture(diameter, center=None):
    '''Makes a Field generator for a circular aperture.

    Parameters
    ----------
    diameter : scalar
        The diameter of the aperture.
    center : array_like
        The center of the aperture

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    def func(grid):
        if grid.is_('cartesian'):
            if grid.is_separated:
                x, y = grid.separated_coords
                x = x[np.newaxis, :]
                y = y[:, np.newaxis]
            else:
                x, y = grid.coords

            f = (((x - shift[0])**2 + (y - shift[1])**2) <= (diameter / 2)**2).ravel()
        else:
            f = grid.as_('polar').r <= (diameter / 2)

        return Field(f.astype('float'), grid)

    return func

def make_elliptical_aperture(diameters, center=None, angle=0):
    '''Makes a Field generator for an elliptical aperture.

    Parameters
    ----------
    diameters : array_like
        The diameters of the aperture in the two directions.
    center : array_like or None
        The center of the aperture.
    angle : scalar
        The orientation of the ellipse in radians.

    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    # Pre-calculate the minor and major axes dimensions after rotation
    major_axis = diameters[0] / 2
    minor_axis = diameters[1] / 2

    cos_angle_major = np.cos(angle) / major_axis
    cos_angle_minor = np.cos(angle) / minor_axis

    sin_angle_major = np.sin(angle) / major_axis
    sin_angle_minor = np.sin(angle) / minor_axis

    def func(grid):
        g = grid.as_('cartesian').shifted(shift)

        if g.is_separated:
            x, y = g.separated_coords
            x = x[np.newaxis, :]
            y = y[:, np.newaxis]
        else:
            x, y = g.coords

        term1 = ((x * cos_angle_major) - (y * sin_angle_major))**2
        term2 = ((x * sin_angle_minor) + (y *  cos_angle_minor))**2
        f = (term1 + term2) <= 1

        return Field(f.ravel().astype('float'), grid)

    return func

def make_rectangular_aperture(size, center=None):
    '''Makes a Field generator for a rectangular aperture.

    Parameters
    ----------
    size : scalar or array_like
        The length of the sides. If this is scalar, a square aperture is assumed.
    center : array_like
        The center of the aperture

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    half_dim = size * np.ones(2) / 2

    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    def func(grid):
        g = grid.as_('cartesian')

        if g.is_separated:
            x, y = g.separated_coords
            x = x[np.newaxis, :]
            y = y[:, np.newaxis]
        else:
            x, y = g.coords

        f = (np.abs(x - shift[0]) <= half_dim[0]) * (np.abs(y - shift[1]) <= half_dim[1])

        return Field(f.ravel().astype('float'), grid)

    return func

def make_irregular_polygon_aperture(vertices):
    '''Make an irregular polygonal aperture.

    Parameters
    ----------
    vertices : iterable of length 2 iterables
        The vertices of the polygon.

    Returns
    -------
    Field generator
        The generator for the polygonal aperture.
    '''
    p = Path(vertices)

    bounding_box_min = np.min(vertices, axis=0)
    bounding_box_max = np.max(vertices, axis=0)

    size = bounding_box_max - bounding_box_min
    center = (bounding_box_min + bounding_box_max) / 2

    def func(grid):
        res = grid.zeros()

        mask = make_rectangular_aperture(size, center=center)(grid).astype('bool')
        res[mask] = p.contains_points(grid.points[mask])

        return res

    return func

def make_regular_polygon_aperture(num_sides, circum_diameter, angle=0, center=None):
    '''Makes a Field generator for a regular-polygon-shaped aperture.

    Parameters
    ----------
    num_sides : integer
        The number of sides for the polygon.
    circum_diameter : scalar
        The circumdiameter of the polygon.
    angle : scalar
        The angle by which to turn the polygon.
    center : array_like
        The center of the aperture

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    if num_sides < 3:
        raise ValueError('The number of sides for a regular polygon has to greater or equal to 3.')

    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    epsilon = 1e-6

    apothem = np.cos(np.pi / num_sides) * circum_diameter / 2
    apothem += apothem * epsilon

    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    # Make use of symmetry
    if num_sides % 2 == 0:
        thetas = np.arange(int(num_sides / 2), dtype='float') * np.pi / int(num_sides / 2) + angle
    else:
        thetas = np.arange(num_sides) * 2 * np.pi / num_sides + angle

    mask = make_rectangular_aperture(circum_diameter)

    def func(grid, return_with_mask=False):
        g = grid.as_('cartesian')

        if g.is_separated:
            x, y = g.separated_coords

            x = x - shift[0]
            y = y - shift[1]

            ind_x = np.flatnonzero(x**2 < ((circum_diameter / 2)**2))
            if not len(ind_x):
                if return_with_mask:
                    return np.array([]), (slice(0, 0), slice(0, 0))
                else:
                    return grid.zeros()

            ind_y = np.flatnonzero(y**2 < ((circum_diameter / 2)**2))
            if not len(ind_y):
                if return_with_mask:
                    return np.array([]), (slice(0, 0), slice(0, 0))
                else:
                    return grid.zeros()

            m_x = slice(ind_x[0], ind_x[-1] + 1)
            m_y = slice(ind_y[0], ind_y[-1] + 1)

            x = x[m_x]
            y = y[m_y]

            f_sub = np.ones((len(ind_y), len(ind_x)))

            if num_sides % 2 == 0:
                for theta in thetas:
                    f_sub *= ((np.cos(theta) * x)[np.newaxis, :] + (np.sin(theta) * y)[:, np.newaxis])**2 <= apothem**2
            else:
                for theta in thetas:
                    f_sub *= ((np.sin(theta) * x)[np.newaxis, :] - (np.cos(theta) * y)[:, np.newaxis]) <= apothem

            if return_with_mask:
                return f_sub, (m_y, m_x)

            f = np.zeros(g.shape)
            f[m_y, m_x] = f_sub

            return Field(f.ravel(), grid)

        # Slow backup method
        m = mask(g) != 0

        x, y = g.coords
        x = x[m] - shift[0]
        y = y[m] - shift[1]

        f_sub = np.ones(x.size, dtype='float')

        # Make use of symmetry
        if num_sides % 2 == 0:
            for theta in thetas:
                f_sub *= (np.cos(theta) * x + np.sin(theta) * y)**2 <= apothem**2
        else:
            for theta in thetas:
                f_sub *= (np.abs(np.sin(theta) * x) + -np.cos(theta) * y) <= apothem

        if return_with_mask:
            return f_sub, m

        f = grid.zeros()
        f[m] = f_sub

        return Field(f, grid)

    return func

# Convenience function
def make_hexagonal_aperture(circum_diameter, angle=0, center=None):
    '''Makes a Field generator for a hexagon aperture.

    Parameters
    ----------
    circum_diameter : scalar
        The circumdiameter of the polygon.
    angle : scalar
        The angle by which to turn the hexagon.
    center : array_like
        The center of the aperture

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    return make_regular_polygon_aperture(6, circum_diameter, angle, center)

def make_spider(p1, p2, spider_width):
    '''Make a rectangular obstruction from `p1` to `p2`.

    Parameters
    ----------
    p1 : list or ndarray
        The starting coordinates of the spider.
    p2 : list or ndarray
        The end coordinates of the spider.
    spider_width : scalar
        The full width of the spider.

    Returns
    -------
    Field generator
        The spider obstruction.
    '''
    delta = np.array(p2) - np.array(p1)
    shift = delta / 2 + np.array(p1)

    spider_angle = np.arctan2(delta[1], delta[0])
    spider_length = np.linalg.norm(delta)

    def func(grid):
        g = grid.as_('cartesian')

        if g.is_separated:
            x, y = g.separated_coords
            x = x[np.newaxis, :]
            y = y[:, np.newaxis]
        else:
            x, y = g.coords

        x = x - shift[0]
        y = y - shift[1]

        x_new = x * np.cos(spider_angle) + y * np.sin(spider_angle)
        y_new = y * np.cos(spider_angle) - x * np.sin(spider_angle)

        # Doing comparisons for each side separately is actually
        # faster than doing abs(x_new) < (spider_length / 2).
        spider = x_new <= (spider_length / 2)
        spider *= x_new >= (-spider_length / 2)
        spider *= y_new <= (spider_width / 2)
        spider *= y_new >= (-spider_width / 2)

        return Field(1 - spider.ravel(), grid)
    return func

def make_spider_infinite(p, angle, spider_width):
    '''Make an infinite spider starting at `p` and extending at an angle `angle`.

    Parameters
    ----------
    p : list or ndarray
        The starting coordinate of the spider.
    angle : scalar
        The angle to which the spider is pointing in degrees.
    spider_width : scalar
        The full width of the spider.

    Returns
    -------
    Field generator
        The spider obstruction.
    '''
    spider_angle = np.radians(angle)

    def func(grid):
        g = grid.as_('cartesian')

        if g.is_separated:
            x, y = g.separated_coords
            x = x[np.newaxis, :]
            y = y[:, np.newaxis]
        else:
            x, y = g.coords

        x = x + p[0]
        y = y + p[1]

        x_new = x * np.cos(spider_angle) + y * np.sin(spider_angle)
        y_new = y * np.cos(spider_angle) - x * np.sin(spider_angle)

        # Doing comparisons for each side separately is actually
        # faster than doing abs(y_new) < (spider_length / 2).
        infinite_spider = y_new <= (spider_width / 2)
        infinite_spider *= y_new >= (-spider_width / 2)
        infinite_spider *= x_new >= 0

        return Field(1 - infinite_spider.ravel(), grid)
    return func

def make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio, num_spiders=0, spider_width=0.01):
    '''Make a simple circular aperture with central obscuration and support structure.

    Parameters
    ----------
    pupil_diameter : scalar
        The diameter of the circular aperture.
    central_obscuration_ratio : scalar
        The ratio of the diameter of the central obscuration compared to the pupil diameter.
    num_spiders : int
        The number of spiders holding up the central obscuration.
    spider_width : scalar
        The full width of the spiders.

    Returns
    -------
    Field generator
        The circularly obstructed aperture.
    '''
    central_obscuration_diameter = pupil_diameter * central_obscuration_ratio

    def func(grid):
        pupil_outer = make_circular_aperture(pupil_diameter)(grid)
        pupil_inner = make_circular_aperture(central_obscuration_diameter)(grid)
        spiders = 1

        spider_angles = np.linspace(0, 2 * np.pi, num_spiders, endpoint=False)

        for angle in spider_angles:
            x = pupil_diameter * np.cos(angle)
            y = pupil_diameter * np.sin(angle)

            spiders *= make_spider((0, 0), (x, y), spider_width)(grid)

        return (pupil_outer - pupil_inner) * spiders
    return func

def make_rotated_aperture(aperture, angle):
    '''Create a rotated version of `aperture`.

    Parameters
    ----------
    aperture : Field generator
        The aperture to rotate.
    angle : scalar
        The angle in radians by which to rotate the aperture.

    Returns
    -------
    Field generator
        The rotated aperture.
    '''
    return lambda grid: Field(aperture(grid.rotated(-angle)), grid)

def make_shifted_aperture(aperture, shift):
    '''Create a shifted version of `aperture`.

    Parameters
    ----------
    aperture : Field generator
        The aperture to rotate.
    shift : array_like
        The shift of the aperture.

    Returns
    -------
    Field generator
        The shifted aperture.
    '''
    return lambda grid: Field(aperture(grid.shifted(-np.array(shift))), grid)

def make_inverted_aperture(aperture):
    '''Create an inverted version of `aperture`.

    Parameters
    ----------
    aperture : Field generator
        The aperture to invert.

    Returns
    -------
    Field generator
        The inverted aperture.
    '''
    return lambda grid: 1 - aperture(grid)

def make_segmented_aperture(segment_shape, segment_positions, segment_transmissions=1, return_segments=False):
    '''Create a segmented aperture.

    Parameters
    ----------
    segment_shape : Field generator
        The shape for each of the segments.
    segment_positions : Grid
        The center position for each of the segments.
    segment_transmissions : scalar or ndarray
        The transmission for each of the segments. If this is a scalar, the same transmission is used for all segments.
    return_segments : boolean
        Whether to return a ModeBasis of all segments as well.

    Returns
    -------
    Field generator
        The segmented aperture.
    list of Field generators
        The segments. Only returned if return_segments is True.
    '''
    segment_transmissions = np.ones(segment_positions.size) * segment_transmissions

    mask_available = 'return_with_mask' in inspect.signature(segment_shape).parameters

    def func(grid):
        res = grid.zeros(dtype=segment_transmissions.dtype)

        for p, t in zip(segment_positions.points, segment_transmissions):
            if mask_available:
                # Use the masked version of the segment shape.
                segment_sub, mask = segment_shape(grid.shifted(-p), return_with_mask=True)

                if isinstance(mask, tuple):
                    res.shaped[mask][segment_sub > 0.5] = t
                else:
                    res[mask][segment_sub > 0.5] = t
            else:
                segment = segment_shape(grid.shifted(-p))
                res[segment > 0.5] = t

        return Field(res, grid)

    if return_segments:
        def seg(grid, p, t):
            return segment_shape(grid.shifted(-p)) * t

        segments = []
        for p, t in zip(segment_positions.points, segment_transmissions):
            segments.append(functools.partial(seg, p=p, t=t))

        return func, segments
    else:
        return func

def make_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size, starting_ring=0, return_segments=False):
    '''Create a hexagonal segmented aperture.

    All segments have a flat-to-flat size of `segment_flat_to_flat`, which means that
    the segment pitch (the distance between the centers of adjacent segments) is the sum
    of the segment flat-to-flat size and the gap size.

    Parameters
    ----------
    num_rings : int
        The number of rings of hexagons to include, not counting the central segment.
        So 2 for a JWST-like aperture, 3 for a Keck-like aperture, and so on.
    segment_flat_to_flat : scalar
        The distance between sides (flat-to-flat) of a single segment.
    gap_size : scalar
        The gap between adjacent segments.
    starting_ring : int
        The first ring of segments. This can be used to exclude the center segment (by
        setting it to one), or the center segment and first ring (by setting it to two).
        The default (zero) includes the center segment.
    return_segments : boolean
        Whether to return a ModeBasis of all segments as well.

    Returns
    -------
    Field generator
        The segmented aperture.
    list of Field generators
        The segments. Only returned if return_segments is True.
    '''
    segment_circum_diameter = segment_flat_to_flat * 2 / np.sqrt(3)
    segment = make_hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    segment_pitch = segment_flat_to_flat + gap_size
    segment_positions = make_hexagonal_grid(segment_pitch, num_rings, pointy_top=False)

    if starting_ring != 0:
        starting_segment = 3 * (starting_ring - 1) * starting_ring + 1

        mask = segment_positions.zeros(dtype='bool')
        mask[starting_segment:] = True

        segment_positions = segment_positions.subset(mask)

    return make_segmented_aperture(segment, segment_positions, return_segments=return_segments)

def make_sector_infinite(p, sector_angle, sector_width):
    """
    Make an infinite sector, starting at `p` and extending at an angle `angle` with a width `angle_width`.

    Parameters
    ----------
    p : list or ndarray
        The center coordinate of the sector.
    sector_angle : scalar
        The angle to which the sector is pointing in degrees [deg].
    sector_width : scalar
        The angular width of the sector [deg].

    Returns
    -------
    Field generator
        The infinite sector.
    """
    sector_angle = np.radians(sector_angle)
    sector_width = np.radians(sector_width)

    def func(grid):
        g = grid.as_('cartesian')   # ABC: now g is grid to be able to perform rotation
        if g.is_separated:
            x, y = g.separated_coords
            x = x[np.newaxis, :]
            y = y[:, np.newaxis]
        else:
            x, y = g.coords

        x = x + p[0]
        y = y + p[1]

        x_new = x * np.cos(sector_angle) + y * np.sin(sector_angle)
        y_new = y * np.cos(sector_angle) - x * np.sin(sector_angle)
        theta = np.arctan2(y_new, x_new)

        infinite_slice = theta <= (sector_width / 2)
        infinite_slice *= theta >= (-sector_width / 2)

        return Field(infinite_slice.ravel(), grid)

    return func

def make_wedge_aperture(inner_diameter, outer_diameter, angle_width, spider_width, pointy_top=True):
    """Create a finite wedge sector, used for keystone aperture.

    Parameters
    ----------
    inner_diameter : scalar
        The inner diameter of the sector, defining the starting boundary of the wedge.
    outer_diameter : scalar
        The outer radius of the sector, defining the end boundary of the wedge.
    angle_width : scalar
        The angular width of the wedge in degrees, specifying the span of the aperture.
    spider_width : scalar
        The full width of the spider.
    pointy_top : bool, optional
        If True the wedge sector will be at the "top", by default True

    Returns
    -------
    Field generator
        The finite wedge sector.
    """
    p = np.zeros(2)
    if pointy_top is False:
        angle = 90
    else:
        angle = 0

    def func(grid):
        sector = make_sector_infinite(p, angle, angle_width)(grid)
        pupil_outer = make_circular_aperture(outer_diameter)(grid)
        pupil_inner = make_circular_aperture(inner_diameter)(grid)
        spider_pos = make_spider_infinite(p, angle + angle_width / 2, spider_width)(grid)
        spider_neg = make_spider_infinite(p, angle - angle_width / 2, spider_width)(grid)

        return (pupil_outer - pupil_inner) * sector * spider_pos * spider_neg

    return func

def make_wedge_annular_aperture(segment_shape, segment_positions, segment_transmissions=1, return_segments=False):
    """Create an annular aperture composed of multiple wedge-shaped segments.

    Parameters
    ----------
    segment_shape : Field generator
        The shape of each individual wedge-shaped segment.
    segment_positions : Grid
        The positions of the centers of the segments in the aperture.
    segment_transmissions : scalar or ndarray, optional
        The transmission for each of the segments. If a scalar is provided, the same
        transmission is applied to all segments. If an array is provided, each element
        specifies the transmission of the corresponding segment. Defaults to 1.
    return_segments : bool, optional
        Whether to return a list of individual segments in addition to the full aperture.
        Defaults to False.

    Returns
    -------
    Field generator
        The annular aperture.
    list of Field generators
        The individual wedge-shaped segments. Only returned if `return_segments` is True.
    """
    segment_transmissions = np.ones(segment_positions.size) * segment_transmissions

    mask_available = 'return_with_mask' in inspect.signature(segment_shape).parameters

    def func(grid):
        res = grid.zeros(dtype=segment_transmissions.dtype)

        for p, t in zip(segment_positions.points, segment_transmissions):
            alpha = np.arctan2(p[1], p[0])
            if mask_available:
                # Use the masked version of the segment shape.
                segment_sub, mask = segment_shape(grid.rotated(-alpha), return_with_mask=True)

                if isinstance(mask, tuple):
                    res.shaped[mask][segment_sub > 0.5] = t
                else:
                    res[mask][segment_sub] = t
            else:
                segment = segment_shape(grid.rotated(-alpha))
                res[segment > 0.5] = t

        return Field(res, grid)

    if return_segments:
        def seg(grid, p, t):
            alpha = np.arctan2(p[1], p[0])
            return segment_shape(grid.rotated(-alpha)) * t

        segments = []
        for p, t in zip(segment_positions.points, segment_transmissions):
            segments.append(functools.partial(seg, p=p, t=t))

        return func, segments
    else:
        return func

def make_keystone_grid(core_diameter, outer_diameter, num_rings, num_keys, pointy_top=True):
    """Make a keystone aperture with a central core diameter.

    Parameters
    ----------
    core_diameter : scalar
        The diameter of the central circular mirror. If 0, no central core is included.
    outer_diameter : scalar
        The total diameter of the aperture, including the outermost ring of wedges.
    num_rings : int
        The number of concentric rings of wedge-shaped segments.
    num_keys : int
        The number of wedge-shaped segments (keys) in each ring.
    pointy_top : bool, optional
        If True, the wedge sector will be at the "top". Defaults to True.

    Returns
    -------
    CartesianGrid
        A Cartesian grid of the keystone aperture's segment positions.
    """
    apothem = outer_diameter * np.sqrt(3) / 4
    segment_pitch = (outer_diameter - core_diameter) / num_rings / 2

    # Generate radial distances and angles
    radii = np.linspace(core_diameter / 2 + segment_pitch / 2, outer_diameter / 2 - segment_pitch / 2, num_rings)
    angles = np.linspace(0, 2 * np.pi, num_keys, endpoint=False) + np.pi / 2

    # Create a grid of polar coordinates
    radial_grid = PolarGrid(SeparatedCoords((radii, angles)))
    cartesian_grid = radial_grid.as_('cartesian')
    x_coords, y_coords = cartesian_grid.coords

    # Add central core position if core_diameter != 0
    if core_diameter != 0:
        x_coords = np.r_[0, x_coords]
        y_coords = np.r_[0, y_coords]

    weight = 2 * apothem**2 * np.sqrt(3)

    if pointy_top:
        return CartesianGrid(UnstructuredCoords((x_coords, y_coords)), weight)
    else:
        return CartesianGrid(UnstructuredCoords((y_coords, x_coords)), weight)

def make_keystone_aperture(core_diameter, outer_diameter, num_rings, radial_gap, num_keys, spider_width, segment_transmissions=1, return_segments=False, pointy_top=True):
    """Create a wedge-shaped keystone aperture.

    The aperture consists of a central circular core and a series of wedge-shaped segments
    arranged in concentric rings. The radial and angular spacing of the segments can be customized
    with the parameters. The aperture can optionally include the individual segments as separate elements.

    Parameters
    ----------
    core_diameter : scalar
        The diameter of the central circular mirror. If 0, no central core is included.
    outer_diameter : scalar
        The total diameter of the aperture, including the outermost ring of wedges.
    num_rings : int
        The number of concentric rings of wedge-shaped segments.
    radial_gap : scalar
        The radial spacing between consecutive rings of wedges.
    num_keys : int
        The number of wedge-shaped segments (keys) in each ring.
    spider_width : scalar
        The width of any central obstructions (e.g., spiders) that intersect the wedges.
    return_segments : bool, optional
        Whether to return a list of individual segments in addition to the full aperture.
        Defaults to False.
    pointy_top : bool, optional
        If True, the wedge sector will be at the "top". Defaults to True.

    Returns
    -------
    Field generator
        The complete wedge-shaped keystone aperture.
    list of Field generators
        The individual wedge segments. Only returned if `return_segments` is True.
    """
    angle_width = 360 / num_keys

    segment_positions = make_keystone_grid(0, outer_diameter, 1, num_keys, pointy_top=True)

    total_segments = num_rings * num_keys + (1 if core_diameter != 0 else 0)
    segment_transmissions = np.ones(total_segments) * segment_transmissions

    def func(grid):
        res = grid.zeros(dtype=segment_transmissions.dtype)
        segment_index = 0

        # Central core
        if core_diameter != 0:
            core_aperture = make_circular_aperture(core_diameter)

            res += core_aperture(grid) * segment_transmissions[0]
            segment_index += 1

        # Rings of keystones
        ring_diameter = np.linspace(core_diameter, outer_diameter, num_rings + 1)

        for i in range(num_rings):
            inner_keystone_diameter = ring_diameter[i] + radial_gap
            outer_keystone_diameter = ring_diameter[i + 1]

            keystone = make_wedge_aperture(inner_keystone_diameter, outer_keystone_diameter, angle_width, spider_width, pointy_top=pointy_top)

            ring_transmissions = segment_transmissions[segment_index:segment_index + num_keys]
            segment_index += num_keys

            keystone_aperture = make_wedge_annular_aperture(keystone, segment_positions, segment_transmissions=ring_transmissions, return_segments=False)

            res += keystone_aperture(grid)

        return Field(res, grid)

    if return_segments:
        segments = []
        segment_index = 0

        # Central core
        if core_diameter != 0:
            core_aperture = make_circular_aperture(core_diameter)

            segments.append(lambda g: core_aperture(g) * segment_transmissions[0])
            segment_index += 1

        # Rings of keystones
        ring_diameter = np.linspace(core_diameter, outer_diameter, num_rings + 1)

        for i in range(num_rings):
            inner_keystone_diameter = ring_diameter[i] + radial_gap
            outer_keystone_diameter = ring_diameter[i + 1]

            keystone = make_wedge_aperture(inner_keystone_diameter, outer_keystone_diameter, angle_width, spider_width, pointy_top=pointy_top)

            ring_transmissions = segment_transmissions[segment_index:segment_index + num_keys]
            segment_index += num_keys

            keystone_aperture = make_wedge_annular_aperture(keystone, segment_positions, segment_transmissions=ring_transmissions, return_segments=return_segments)
            _, _segments = keystone_aperture

            segments.extend(_segments)

        return func, segments
    else:
        return func

def make_super_gaussian_aperture(full_width_half_maximum, shape_parameter=5, center=None):
    '''Make a super Guassian aperture.

    Parameters
    ----------
    full_width_half_maximum : scalar
        The Full Width Half Maximum of the super Gaussian.
    shape_parameter : scalar
        The shape parameter determines the exponential Gaussian fall off. p=1 is identical to a normal Gaussian and p=0.5 results in a Laplacian distribution.

    Returns
    -------
    Field generator
        The super Gaussian aperture.
    '''
    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)

    def func(grid):
        sigma = 0.5 * full_width_half_maximum / (2 * np.log(2))**(1 / (2 * shape_parameter))
        r = grid.shifted(-shift).as_('polar').r / sigma
        return Field(np.exp(-1 / 2 * abs(r)**(2 * shape_parameter)), grid)

    return func

@deprecated_name_changed(make_circular_aperture)
def circular_aperture():
    pass

@deprecated_name_changed(make_elliptical_aperture)
def elliptical_aperture():
    pass

@deprecated_name_changed(make_rectangular_aperture)
def rectangular_aperture():
    pass

@deprecated_name_changed(make_hexagonal_aperture)
def hexagonal_aperture():
    pass

@deprecated_name_changed(make_regular_polygon_aperture)
def regular_polygon_aperture():
    pass

@deprecated_name_changed(make_irregular_polygon_aperture)
def irregular_polygon_aperture():
    pass

@deprecated_name_changed(make_inverted_aperture)
def make_obstruction():
    pass
