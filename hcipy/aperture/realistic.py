import numpy as np

from ..field import make_hexagonal_grid, Field
from .generic import make_elliptical_aperture, make_spider, make_circular_aperture, make_hexagonal_aperture, make_segmented_aperture, make_shifted_aperture, make_spider_infinite, make_obstructed_circular_aperture, make_rectangular_aperture, make_obstruction, make_regular_polygon_aperture, make_irregular_polygon_aperture

import functools

_vlt_telescope_aliases = {'antu': 'ut1', 'kueyen': 'ut2', 'melipal': 'ut3', 'yepun': 'ut4'}

def make_vlt_aperture(
        normalized=False, telescope='ut3', with_spiders=True, with_M3_cover=False,
        return_segments=False):
    '''Make the VLT aperture.

    This aperture is based on the ERIS pupil documentation: VLT-SPE-AES-11310-0006.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 8.0 meters for UT1-3 and 8.1196 meters for UT4.
    telescope : one of {'ut1', 'ut2', 'ut3', 'ut4', 'antu', 'kueyen', 'melipal', 'yepun'}
        The specific telescope on the VLT, case insensitive. Default: UT3.
    with_spiders : boolean
        If this is False, the spiders will be left out. Default: True.
    with_M3_cover : boolean
        If this is True, a cover will be created for the M3 in stowed position.
        This M3 cover is only available on UT4, mimicking the ERIS pupil. A warning
        will be emitted when using an M3 cover with other UTs. Default: False.
    return_segments : boolean
        If this is True, the pupil quadrants (segments) will also be returned.

    Returns
    -------
    aperture : Field generator
        The VLT aperture.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    telescope = telescope.lower()
    if telescope not in ['ut1', 'ut2', 'ut3', 'ut4']:
        if telescope not in _vlt_telescope_aliases:
            raise ValueError(f'The VLT telescope "{telescope}" has not been implemented.')
        telescope = _vlt_telescope_aliases[telescope]

    if telescope in ['ut1', 'ut2', 'ut3']:
        pupil_diameter = 8.0  # meter
        central_obscuration_ratio = 1.116 / pupil_diameter
    elif telescope == 'ut4':
        pupil_diameter = 8.1196  # meter
        central_obscuration_ratio = 0.6465 * 2 / pupil_diameter

    spider_width = 0.040  # meter
    spider_offset = 0.4045  # meter
    spider_outer_radius = 4.2197  # meter
    outer_diameter_M3_stow = 1.070  # meter
    angle_between_spiders = 101  # degrees

    if with_M3_cover and telescope != 'ut4':
        import warnings
        warnings.warn('Using the M3 cover on a telescope other than UT4 is not realistic.', stacklevel=2)

    if normalized:
        spider_width /= pupil_diameter
        spider_offset /= pupil_diameter
        spider_outer_radius /= pupil_diameter
        outer_diameter_M3_stow /= pupil_diameter
        pupil_diameter = 1.0

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)

    if with_spiders or return_segments:
        spider_inner_radius = spider_offset / np.cos(np.radians(45 - (angle_between_spiders - 90) / 2))

        spider_start_1 = -spider_inner_radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        spider_end_1 = spider_outer_radius * np.array([np.cos(np.pi), np.sin(np.pi)])

        spider_start_2 = -spider_inner_radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        spider_end_2 = spider_outer_radius * np.array([np.cos(-np.pi / 2), np.sin(-np.pi / 2)])

        spider_start_3 = spider_inner_radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        spider_end_3 = spider_outer_radius * np.array([np.cos(0), np.sin(0)])

        spider_start_4 = spider_inner_radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        spider_end_4 = spider_outer_radius * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])

    if with_spiders:
        spider1 = make_spider(spider_start_1, spider_end_1, spider_width)
        spider2 = make_spider(spider_start_2, spider_end_2, spider_width)
        spider3 = make_spider(spider_start_3, spider_end_3, spider_width)
        spider4 = make_spider(spider_start_4, spider_end_4, spider_width)

    if with_M3_cover:
        m3_cover = make_obstruction(make_rectangular_aperture(outer_diameter_M3_stow, center=[outer_diameter_M3_stow / 2, 0]))

    if with_spiders:
        if with_M3_cover:
            def func(grid):
                return Field(obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * m3_cover(grid), grid)
        else:
            def func(grid):
                return Field(obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid), grid)
    else:
        if with_M3_cover:
            def func(grid):
                return Field(obstructed_aperture(grid) * m3_cover(grid), grid)
        else:
            def func(grid):
                return Field(obstructed_aperture(grid), grid)

    if return_segments:
        n1 = (spider_end_1[1] - spider_start_1[1], spider_start_1[0] - spider_end_1[0])
        c1 = np.dot(n1, spider_end_1)

        n2 = (spider_end_2[1] - spider_start_2[1], spider_start_2[0] - spider_end_2[0])
        c2 = np.dot(n2, spider_end_2)

        n3 = (spider_end_3[1] - spider_start_3[1], spider_start_3[0] - spider_end_3[0])
        c3 = np.dot(n3, spider_end_3)

        n4 = (spider_end_4[1] - spider_start_4[1], spider_start_4[0] - spider_end_4[0])
        c4 = np.dot(n4, spider_end_4)

        ns = [n1, n4, n3, n2]
        cs = [c1, c4, c3, c2]

        def segment(n1, c1, n2, c2, grid):
            if grid.is_separated:
                x, y = grid.separated_coords
                f = (np.dot(n1, np.array([x[np.newaxis, :], y[:, np.newaxis]], dtype = object)) > c1) * 1.0
                f *= (np.dot(n2, np.array([x[np.newaxis, :], y[:, np.newaxis]], dtype = object)) < c2) * 1.0
                intersection = np.array([c1, c2]).dot(np.linalg.inv(np.array([n1, n2])))
                ni = np.array([-intersection[1], -intersection[0]])
                f *= (np.dot(ni, np.array([x[np.newaxis, :], y[:, np.newaxis]], dtype = object)) < 0) * 1.0
            else:
                x, y = grid.coords
                f = (np.dot(n1, np.array([x, y])) > c1) * 1.0
                f *= (np.dot(n2, np.array([x, y])) < c2) * 1.0
                intersection = np.array([c1, c2]).dot(np.linalg.inv(np.array([n1, n2])))
                ni = np.array([-intersection[1], -intersection[0]])
                f *= (np.dot(ni, np.array([x, y])) < 0) * 1.0
            f = f.ravel()
            f *= func(grid)

            if with_M3_cover:
                f *= m3_cover(grid)
            return Field(f.astype('float'), grid)

        segments = []
        for i in range(4):
            segments.append(functools.partial(
                segment, np.roll(ns, -i, axis=0)[0], np.roll(cs, -i, axis=0)[0],
                np.roll(ns, -i, axis=0)[1], np.roll(cs, -i, axis=0)[1]))

    if return_segments:
        return func, segments
    else:
        return func

def make_vlti_aperture(zenith_angle=0, azimuth=0, with_spiders=True, return_segments=False):
    '''Make the VLTI aperture for interferometry.

    The position of each VLT is taken from the VLTI user manual: VLT-MAN-ESO-15000-4552.
    This function does not apply any differential piston.

    Parameters
    ----------
    zenith_angle : scalar
        The zenith angle component of the pointing direction.
    azimuth : scalar
        The azimuth component of the pointing direction. Defined from north to east.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    return_segments : boolean
        If this is True, the segments will also be returned as a ModeBasis.

    Returns
    -------
    Field generator
        The VLTI aperture.
    segments : list of Field generators
        The individual telescopes. Only returned when `return_segments` is True.
    '''
    # UT1 is taken as a reference
    baseline_UT12 = np.array([24.8, 50.8])
    baseline_UT13 = np.array([54.8, 86.5])
    baseline_UT14 = np.array([113.2, 64.3])

    relative_position = np.array([[0, 0], baseline_UT12, baseline_UT13, baseline_UT14])

    # Calculate the middle between the extremes of the 4 telescopes
    reference_position = (np.max(relative_position, axis=0) + np.min(relative_position, axis=0)) / 2
    telescope_positions = np.array([-reference_position, baseline_UT12 - reference_position, baseline_UT13 - reference_position, baseline_UT14 - reference_position])

    # Make the new local coordinate basis from the pointing vector
    def make_point_vector(theta, phi):
        return np.array([np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)])

    w = make_point_vector(zenith_angle, azimuth)
    v = make_point_vector(zenith_angle + np.pi / 2, azimuth)
    u = np.cross(v, w)

    telescope_apertures = []
    for telescope_name, position in zip(['ut1', 'ut2', 'ut3', 'ut4'], telescope_positions):
        u_position = u[0] * position[0] + u[1] * position[1]
        v_position = v[0] * position[0] + v[1] * position[1]

        single_ut = make_shifted_aperture(make_vlt_aperture(telescope=telescope_name, with_spiders=with_spiders), shift=[u_position, v_position])
        telescope_apertures.append(single_ut)

    def func(grid):
        res = 0
        for ap in telescope_apertures:
            res += ap(grid)
        return res

    if return_segments:
        return func, telescope_apertures
    else:
        return func

def make_vlti_dopd_map(zenith_angle=0, azimuth=0, with_spiders=True, return_segments=False):
    '''Make the VLTI differential OPD map for interferometry.

    The position of each VLT is taken from the VLTI user manual: VLT-MAN-ESO-15000-4552.

    Parameters
    ----------
    zenith_angle : scalar
        The zenith angle component of the pointing direction.
    azimuth : scalar
        The azimuth component of the pointing direction. Defined from north to east.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    return_segments : boolean
        If this is True, the segments will also be returned as a ModeBasis.

    Returns
    -------
    Field generator
        The full VLTI dOPD map.
    segments : list of Field generators
        The individual dOPD maps for each telescope. Only returned when `return_segments` is True.
    '''
    # UT1 is taken as a reference
    baseline_UT12 = np.array([24.8, 50.8])
    baseline_UT13 = np.array([54.8, 86.5])
    baseline_UT14 = np.array([113.2, 64.3])

    relative_position = np.array([[0, 0], baseline_UT12, baseline_UT13, baseline_UT14])

    # Calculate the middle between the extremes of the 4 telescopes
    reference_position = (np.max(relative_position, axis=0) + np.min(relative_position, axis=0)) / 2
    telescope_positions = np.array([-reference_position, baseline_UT12 - reference_position, baseline_UT13 - reference_position, baseline_UT14 - reference_position])

    # Make the new local coordinate basis from the pointing vector
    def make_point_vector(theta, phi):
        return np.array([np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)])

    w = make_point_vector(zenith_angle, azimuth)
    v = make_point_vector(zenith_angle + np.pi / 2, azimuth)
    u = np.cross(v, w)

    dOPD = []
    telescope_apertures = []
    for telescope_name, position in zip(['ut1', 'ut2', 'ut3', 'ut4'], telescope_positions):
        u_position = u[0] * position[0] + u[1] * position[1]
        v_position = v[0] * position[0] + v[1] * position[1]
        dOPD_i = w[0] * position[0] + w[1] * position[1]
        dOPD.append(dOPD_i)

        single_ut = make_shifted_aperture(make_vlt_aperture(telescope=telescope_name, with_spiders=with_spiders), shift=[u_position, v_position])

        telescope_apertures.append(single_ut)

    def ut_with_opd(ut, dOPD_i):
        return lambda grid: ut(grid) * dOPD_i
    telescope_apertures = [ut_with_opd(ut, dOPD_i) for ut, dOPD_i in zip(telescope_apertures, dOPD)]

    def func(grid):
        res = 0
        for ap in telescope_apertures:
            res += ap(grid)
        return res

    if return_segments:
        return func, telescope_apertures
    else:
        return func

def make_subaru_aperture():
    pass

def make_lbt_aperture():
    pass

def make_magellan_aperture(normalized=False, with_spiders=True):
    '''Make the Magellan aperture.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 6.5 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Magellan aperture.
    '''
    pupil_diameter = 6.5  # meter
    spider_width1 = 0.75 * 0.0254  # meter
    spider_width2 = 1.5 * 0.0254  # meter
    central_obscuration_ratio = 0.29
    spider_offset = [0, 0.34]  # meter

    if normalized:
        spider_width1 /= pupil_diameter
        spider_width2 /= pupil_diameter
        spider_offset = [x / pupil_diameter for x in spider_offset]
        pupil_diameter = 1.0

    spider_offset = np.array(spider_offset)

    mirror_edge1 = (pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge2 = (-pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge3 = (pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge4 = (-pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)

    if not with_spiders:
        return obstructed_aperture

    spider1 = make_spider(spider_offset, mirror_edge1, spider_width1)
    spider2 = make_spider(spider_offset, mirror_edge2, spider_width1)
    spider3 = make_spider(-spider_offset, mirror_edge3, spider_width2)
    spider4 = make_spider(-spider_offset, mirror_edge4, spider_width2)

    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    return func

def make_hale_aperture(normalized=False, with_spiders=True):
    '''Make the Hale Telescope aperture.

    The size of the spiders and the attachements were based on the actual pupil measurements from Figure 2 of [Galicher2019]_

    .. [Galicher2019] Raphael Galicher et al., "Minimization of non-common path aberrations at the Palomar telescope using a self-coherent camera." Astronomy & Astrophysics 631 (2019): A143.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 5.08 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Hale Telescope aperture.
    '''
    pupil_diameter = 5.08  # meter
    central_obscuration_diameter = 1.86  # meter
    spider_width = 2 * 0.024  # meter
    central_obscuration_ratio = central_obscuration_diameter / pupil_diameter

    # Attachement points of the spiders at the central obscuration
    box_heigth = 2 * 0.06
    box_width = 2 * 0.0932 + central_obscuration_diameter

    if normalized:
        spider_width /= pupil_diameter
        box_heigth /= pupil_diameter
        box_width /= pupil_diameter
        pupil_diameter = 1.0

    box1 = make_rectangular_aperture([box_width, box_heigth])
    box2 = make_rectangular_aperture([box_heigth, box_width])

    if not with_spiders:
        obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)
    else:
        obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio, num_spiders=4, spider_width=spider_width)

    def func(grid):
        return Field(obstructed_aperture(grid) * (1 - box1(grid)) * (1 - box2(grid)), grid)

    return func

def make_luvoir_a_aperture(
        normalized=False, with_spiders=True, with_segment_gaps=True,
        gap_padding=1, segment_transmissions=1, return_header=False, return_segments=False):
    '''Make the LUVOIR A aperture.

    This aperture changes frequently. This one is based on LUVOIR Apertures dimensions
    from Matt Bolcar, LUVOIR lead engineer (as of 10 April 2019)
    Spiders and segment gaps can be included or excluded, and the transmission for each
    of the segments can also be changed. Segments can be returned as well.

    Parameters
    ----------
    normalized : boolean
        If this is True, the pupil diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    gap_padding : scalar
        Arbitrary padding of gap size to represent gaps on smaller arrays - this effectively
        makes the gaps larger and the segments smaller to preserve the same segment pitch.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission
        will be used for all segments.
    return_header : boolean
        If this is True, a header will be returned giving all important values for the
        created aperture for reference.
    return_segments : boolean
        If this is True, the segments will also be returned as a ModeBasis.

    Returns
    -------
    aperture : Field generator
        The LUVOIR A aperture.
    aperture_header : dict
        A dictionary containing all quantities used when making this aperture. Only returned if
        `return_header` is True.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    pupil_diameter = 15.0  # m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
    pupil_inscribed = 13.5  # m actual inscribed diameter
    actual_segment_flat_diameter = 1.2225  # m actual segment flat-to-flat diameter
    actual_segment_gap = 0.006  # m actual gap size between segments
    spider_width = 0.150  # m actual strut size
    spid_start = 0.30657  # m spider starting point distance from center of aperture
    num_rings = 6  # number of full rings of hexagons around central segment
    lower_spider_angle = 12.7  # deg spiders are upside-down 'Y' shaped; degree the lower two spiders are offset from vertical by this amount

    # padding out the segmentation gaps so they are visible and not sub-pixel
    segment_gap = actual_segment_gap * gap_padding
    if not with_segment_gaps:
        segment_gap = 0

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter

    if not with_segment_gaps:
        segment_gap = 0

    aperture_header = {
        'TELESCOP': 'LUVOIR A',
        'D_CIRC': pupil_diameter,
        'D_INSC': pupil_inscribed,
        'SEG_F2F_D': actual_segment_flat_diameter,
        'SEG_GAP': actual_segment_gap,
        'STRUT_W': spider_width,
        'STRUT_AN': lower_spider_angle,
        'NORM': normalized,
        'SEG_TRAN': segment_transmissions,
        'GAP_PAD': gap_padding,
        'STRUT_ST': spid_start,
        'PROV': 'MBolcar ppt 20180815'
    }

    if normalized:
        segment_circum_diameter /= pupil_diameter
        actual_segment_flat_diameter /= pupil_diameter
        actual_segment_gap /= pupil_diameter
        spider_width /= pupil_diameter
        spid_start /= pupil_diameter
        pupil_diameter = 1.0

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)

    # clipping the "corner" segments of the outermost rings
    segment_positions = segment_positions.subset(make_circular_aperture(pupil_diameter * 0.98))
    segment_positions = segment_positions.subset(lambda grid: ~(make_circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = make_hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    if with_spiders:
        spider1 = make_spider_infinite([0, 0], 90, spider_width)
        spider2 = make_spider_infinite([spid_start, 0], 270 - lower_spider_angle, spider_width)
        spider3 = make_spider_infinite([-spid_start, 0], 270 + lower_spider_angle, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=return_segments)

    if return_segments:
        segmented_aperture, segments = segmented_aperture

    def func(grid):
        res = segmented_aperture(grid)

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid)

        return Field(res, grid)

    if with_spiders and return_segments:
        # Use function to return the lambda, to avoid incorrect binding of variables
        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid)

        segments = [segment_with_spider(s) for s in segments]

    if return_header:
        if return_segments:
            return func, aperture_header, segments
        else:
            return func, aperture_header
    elif return_segments:
        return func, segments
    else:
        return func

def make_luvoir_a_lyot_stop(
        normalized=False, with_spiders=False, spider_oversize=1,
        inner_diameter_fraction=0.2, outer_diameter_fraction=0.9, return_header=False):
    '''Make a LUVOIR-A Lyot stop for the APLC coronagraph.

    Parameters
    ----------
    normalized : boolean
        If this is True, the pupil diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    inner_diameter_fraction : scalar
        The fractional size of the circular central obstruction as fraction of the pupil diameter.
    outer_diameter_fraction : scalar
        The fractional size of the circular outer edge as fraction of the pupil diameter.
    spider_oversize : scalar
        The factor by which to oversize the spiders compared to the LUVOIR-A aperture spiders.
    return_header : boolean
        If this is True, a header will be returned giving all important values for the
        created aperture for reference.

    Returns
    -------
    lyot_stop : Field generator
        A field generator for the Lyot stop.
    header : dict
        A dictionary containing all important values for the created aperture. Only returned
        if `return_header` is True.
    '''
    pupil_diameter = 15.0  # m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
    spider_width = 0.150  # m actual strut size
    lower_spider_angle = 12.7  # deg angle at which lower spiders are offset from vertical
    spid_start = 0.30657  # m spider starting point offset from center of aperture

    outer_D = pupil_diameter * outer_diameter_fraction
    inner_D = pupil_diameter * inner_diameter_fraction
    pad_spid_width = spider_width * spider_oversize

    lyot_reference_diameter = pupil_diameter

    ls_header = {
        'TELESCOP': 'LUVOIR A',
        'D_CIRC': pupil_diameter,
        'D_INSC': 13.5,
        'LS_ID': inner_diameter_fraction,
        'LS_OD': outer_diameter_fraction,
        'LS_REF_D': lyot_reference_diameter,
        'NORM': normalized,
        'STRUT_ST': spid_start
    }

    if with_spiders:
        ls_header['STRUT_W']  = spider_width
        ls_header['STRUT_AN'] = lower_spider_angle
        ls_header['STRUT_P']  = spider_oversize

    if normalized:
        outer_D /= pupil_diameter
        inner_D /= pupil_diameter
        pad_spid_width /= pupil_diameter
        spid_start /= pupil_diameter

    outer_diameter = make_circular_aperture(outer_D)
    central_obscuration = make_circular_aperture(inner_D)

    if with_spiders:
        spider1 = make_spider_infinite([0, 0], 90, pad_spid_width)
        spider2 = make_spider_infinite([spid_start, 0], 270 - lower_spider_angle, pad_spid_width)
        spider3 = make_spider_infinite([-spid_start, 0], 270 + lower_spider_angle, pad_spid_width)

    def aper(grid):
        result = outer_diameter(grid) - central_obscuration(grid)

        if with_spiders:
            result *= spider1(grid) * spider2(grid) * spider3(grid)

        return result

    if return_header:
        return aper, ls_header

    return aper

def make_luvoir_b_aperture(
        normalized=False, with_segment_gaps=True, gap_padding=1, segment_transmissions=1,
        return_header=False, return_segments=False):
    '''Make the LUVOIR B aperture.

    This aperture is based on the LUVOIR final report 2019.
    Segment gaps can be included or excluded, and the transmission for each
    of the segments can also be changed. Segments can be returned as well.

    Parameters
    ----------
    normalized : boolean
        If this is True, the pupil diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 8.0 meters.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    gap_padding : scalar
        Arbitrary padding of gap size to represent gaps on smaller arrays - this effectively
        makes the gaps larger and the segments smaller to preserve the same segment pitch.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission
        will be used for all segments.
    return_header : boolean
        If this is True, a header will be returned giving all important values for the
        created aperture for reference.
    return_segments : boolean
        If this is True, the segments will also be returned as a ModeBasis.

    Returns
    -------
    aperture : Field generator
        The LUVOIR B aperture.
    aperture_header : dict
        A dictionary containing all quantities used when making this aperture. Only returned if
        `return_header` is True.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    pupil_diameter = 8.0  # m actual circumscribed diameter, used for lam/D calculations other measurements normalized by this diameter
    pupil_inscribed = 6.7  # m actual inscribed diameter
    actual_segment_flat_diameter = 0.955  # m actual segment flat-to-flat diameter
    actual_segment_gap = 0.006  # m actual gap size between segments

    # padding out the segmentation gaps so they are visible and not sub-pixel
    segment_gap = actual_segment_gap * gap_padding
    if not with_segment_gaps:
        segment_gap = 0

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter

    num_rings = 4  # number of full rings of hexagons around central segment

    if not with_segment_gaps:
        segment_gap = 0

    aperture_header = {
        'TELESCOP': 'LUVOIR B',
        'D_CIRC': pupil_diameter,
        'D_INSC': pupil_inscribed,
        'SEG_F2F_D': actual_segment_flat_diameter,
        'SEG_GAP': actual_segment_gap,
        'NORM': normalized,
        'SEG_TRAN': segment_transmissions,
        'GAP_PAD': gap_padding,
        'PROV': 'LUVOIR final report 2019'
    }

    if normalized:
        segment_circum_diameter /= pupil_diameter
        actual_segment_flat_diameter /= pupil_diameter
        actual_segment_gap /= pupil_diameter
        pupil_diameter = 1.0

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)

    # Clipping the "corner" segments of the outermost rings
    segment_positions = segment_positions.subset(make_circular_aperture(pupil_diameter * 0.9))

    segment = make_hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=return_segments)

    if return_segments:
        segmented_aperture, segments = segmented_aperture

    def func(grid):
        res = segmented_aperture(grid)

        return Field(res, grid)

    if return_header:
        if return_segments:
            return func, aperture_header, segments
        else:
            return func, aperture_header
    elif return_segments:
        return func, segments
    else:
        return func

def make_hicat_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, return_header=False, return_segments=False):
    '''Make the HiCAT P3 apodizer mask

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    return_header : boolean
        If this is True, a header will be returned giving all important values for the
        created aperture for reference.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    aperture : Field generator
        The HiCAT aperture.
    header : dict
        A dictionary containing all important values for the created aperture. Only returned
        if `return_header` is True.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    gamma_21 = 0.423
    gamma_31 = 1.008

    # P2 - Iris AO
    p2_irisao_segment_size = 1.4e-3  # m (note: point to point)
    p2_irisao_segment_side_length = p2_irisao_segment_size / 2
    p2_irisao_segment_gap_size = 12e-6  # m

    p2_irisao_distance_between_segments = p2_irisao_segment_side_length * np.sqrt(3)
    p2_irisao_segment_circumdiameter = (2 * p2_irisao_segment_side_length) - (2 / np.sqrt(3)) * p2_irisao_segment_gap_size

    # P3 - Apodizer
    # Contour
    p3_apodizer_size = 19.725e-3  # m

    # Gap
    p3_apodizer_mask_gap_size = 0.090e-3  # m
    p3_irisao_segment_gap_size = p2_irisao_segment_gap_size * gamma_31 / gamma_21
    apodizer_mask_gap_oversize_factor_wrt_irisao = p3_apodizer_mask_gap_size / p3_irisao_segment_gap_size

    # Central segment
    p3_apodizer_mask_central_segment_size = 3.950e-3  # m

    # Spiders
    p3_apodizer_mask_spiders_thickness = 0.350e-3  # m

    header = {
        'TELESCOP': 'HiCAT',
        'P3_APOD': p3_apodizer_size,
        'P3_CENT_SEG': p3_apodizer_mask_central_segment_size,
        'P3_GAP': p3_apodizer_mask_gap_size,
        'P3_GAP_OVER': apodizer_mask_gap_oversize_factor_wrt_irisao,
        'P3_STRUT': p3_apodizer_mask_spiders_thickness,
        'PROV': 'HiCAT spreadsheet'
    }

    p3_irisao_segment_circumdiameter = p2_irisao_segment_circumdiameter * gamma_31 / gamma_21
    p3_irisao_distance_between_segments = p2_irisao_distance_between_segments * gamma_31 / gamma_21
    p3_apodizer_segment_circumdiameter = p3_irisao_segment_circumdiameter + (-p3_apodizer_mask_gap_size + p3_irisao_segment_gap_size) * (2 / np.sqrt(3))

    if normalized:
        p3_apodizer_segment_circumdiameter /= p3_apodizer_size
        p3_irisao_distance_between_segments /= p3_apodizer_size
        p3_apodizer_mask_central_segment_size /= p3_apodizer_size
        p3_apodizer_mask_spiders_thickness /= p3_apodizer_size
        p3_apodizer_size = 1

    segment = make_hexagonal_aperture(p3_apodizer_segment_circumdiameter, np.pi / 2)
    segment_positions = make_hexagonal_grid(p3_irisao_distance_between_segments, 3, False)
    segmentation = make_segmented_aperture(segment, segment_positions)

    segment = make_hexagonal_aperture(p3_apodizer_size / 7 / np.sqrt(3) * 2, np.pi / 2)
    distance_between_segments = p3_apodizer_size / 7
    segment_positions = make_hexagonal_grid(distance_between_segments, 3)
    contour = make_segmented_aperture(segment, segment_positions, return_segments=return_segments)

    if return_segments:
        contour, segments = contour

    central_segment = make_hexagonal_aperture(p3_apodizer_mask_central_segment_size, np.pi / 2)

    if with_spiders:
        spider1 = make_spider_infinite([0, 0], 60, p3_apodizer_mask_spiders_thickness)
        spider2 = make_spider_infinite([0, 0], 120, p3_apodizer_mask_spiders_thickness)
        spider3 = make_spider_infinite([0, 0], -60, p3_apodizer_mask_spiders_thickness)
        spider4 = make_spider_infinite([0, 0], -120, p3_apodizer_mask_spiders_thickness)

    def func(grid):
        res = contour(grid) - central_segment(grid)

        if with_segment_gaps:
            res *= segmentation(grid)

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)

        return Field(res, grid)

    if return_segments:
        def segment_with_obstructions(seg):
            def func2(grid):
                return Field(func(grid) * seg(grid), grid)

            return func2

        segments = [segment_with_obstructions(s) for s in segments]

    if return_header:
        if return_segments:
            return func, header, segments
        else:
            return func, header
    elif return_segments:
        return func, segments
    else:
        return func

def make_hicat_lyot_stop(normalized=False, with_spiders=True, inner_diameter_fraction=0.2, outer_diameter_fraction=0.9, return_header=False):
    '''Make the HiCAT Lyot stop.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    inner_diameter_fraction : scalar
        The fractional size of the circular central obstruction as fraction of the pupil diameter.
    outer_diameter_fraction : scalar
        The fractional size of the circular outer edge as fraction of the pupil diameter.
    return_header : boolean
        If this is True, a header will be returned giving all important values for the
        created aperture for reference.

    Returns
    -------
    lyot_stop : Field generator
        A field generator for the Lyot stop.
    header : dict
        A dictionary containing all important values for the created aperture. Only returned
        if `return_header` is True.
    '''
    gamma_31 = 1.008
    gamma_51 = 0.979

    p3_apodizer_mask_central_segment_size = 3.950e-3  # m
    p3_apodizer_size = 19.725e-3  # m
    p5_apodizer_size = p3_apodizer_size * gamma_51 / gamma_31

    p5_lyot_stop_size = outer_diameter_fraction * p5_apodizer_size  # m

    # Central segment
    p5_lyot_stop_mask_central_segment_size = inner_diameter_fraction * p5_apodizer_size  # m

    # Spiders
    p5_lyot_stop_mask_spiders_thickness = 0.700e-3  # m

    if normalized:
        p5_lyot_stop_size /= p5_apodizer_size
        p5_lyot_stop_mask_central_segment_size /= p5_apodizer_size
        p5_lyot_stop_mask_spiders_thickness /= p5_apodizer_size

    central_obscuration = make_circular_aperture(p5_lyot_stop_mask_central_segment_size)
    outer_diameter = make_circular_aperture(p5_lyot_stop_size)

    header = {
        'TELESCOP': 'HiCAT',
        'P3_APOD': p3_apodizer_size,
        'P3_CENT_SEG': p3_apodizer_mask_central_segment_size,
        'LS_CENT': p5_lyot_stop_mask_central_segment_size,
        'LS_SIZE': p5_lyot_stop_size,
        'P5_STRUT': p5_lyot_stop_mask_spiders_thickness
    }

    if with_spiders:
        spider1 = make_spider_infinite([0, 0], 60, p5_lyot_stop_mask_spiders_thickness)
        spider2 = make_spider_infinite([0, 0], 120, p5_lyot_stop_mask_spiders_thickness)
        spider3 = make_spider_infinite([0, 0], -60, p5_lyot_stop_mask_spiders_thickness)
        spider4 = make_spider_infinite([0, 0], -120, p5_lyot_stop_mask_spiders_thickness)

    def func(grid):
        res = outer_diameter(grid) - central_obscuration(grid)

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)

        return Field(res, grid)

    if return_header:
        return func, header
    else:
        return func

def make_elt_aperture(normalized=False, with_spiders=True, segment_transmissions=1, return_segments=False):
    '''Make the European Extremely Large Telescope aperture.

    This aperture is based on Figure 3.66 that describes the pupil in the E-ELT Construction Proposal:
        https://www.eso.org/sci/facilities/eelt/docs/index.html .

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 39.14634 meters.
    with_spiders : boolean
        If this is False, the spiders will be left out. Default: True.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission
        will be used for all segments.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    Field generator
        The E-ELT aperture.
    elt_segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    elt_outer_diameter = 39.14634
    spider_width = 0.4
    segment_size = 1.45
    segment_gap = 0.004
    inner_diameter = 9.4136
    outer_diameter = 39.14634

    if normalized:
        segment_size /= elt_outer_diameter
        segment_gap /= elt_outer_diameter
        inner_diameter /= elt_outer_diameter
        outer_diameter /= elt_outer_diameter
        spider_width /= elt_outer_diameter

    segment_positions = make_hexagonal_grid(segment_size * np.sqrt(3) / 2 + segment_gap, 17, pointy_top=False)

    # remove the inner segments
    central_obscuration_mask = (1 - make_hexagonal_aperture(inner_diameter * 2 / np.sqrt(3))(segment_positions)) > 0
    segment_positions = segment_positions.subset(central_obscuration_mask)

    # remove the pointy tops for a more circular aperture
    edge_mask_top = abs(segment_positions.y) < ((outer_diameter / 2) * 0.99)
    edge_mask_positive_30 = abs(np.cos(np.pi / 6) * segment_positions.x + np.sin(np.pi / 6) * segment_positions.y) < ((outer_diameter / 2) * 0.99)
    edge_mask_negative_30 = abs(np.cos(np.pi / 6) * segment_positions.x - np.sin(np.pi / 6) * segment_positions.y) < ((outer_diameter / 2) * 0.99)
    all_edge_masks = edge_mask_top * edge_mask_positive_30 * edge_mask_negative_30 > 0

    segment_positions = segment_positions.subset(all_edge_masks)

    segment_shape = make_hexagonal_aperture(segment_size, angle=np.pi / 2)

    if return_segments:
        elt_aperture_function, elt_segments = make_segmented_aperture(segment_shape, segment_positions, segment_transmissions, return_segments=return_segments)
    else:
        elt_aperture_function = make_segmented_aperture(segment_shape, segment_positions, segment_transmissions)

    spiders = [make_spider_infinite([0, 0], 60 * i + 30, spider_width) for i in range(6)]

    def elt_aperture_with_spiders(grid):
        aperture = elt_aperture_function(grid)

        if with_spiders:
            for spider in spiders:
                aperture *= spider(grid)

        return aperture

    if with_spiders and return_segments:
        # Use function to return the lambda, to avoid incorrect binding of variables
        def spider_func(grid):
            spider_aperture = grid.ones()
            for spider in spiders:
                spider_aperture *= spider(grid)
            return spider_aperture

        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider_func(grid)

        elt_segments = [segment_with_spider(s) for s in elt_segments]

    if return_segments:
        return elt_aperture_with_spiders, elt_segments
    else:
        return elt_aperture_with_spiders

def make_gmt_aperture(normalized=False, with_spiders=True, return_segments=False):
    '''Make the Giant Magellan Telescope aperture.

    The primary mirror parameters come from the GMT Observatory Architecture Documentation (GMT-REQ-03215, Rev. F):
    https://www.gmto.org/resources/slpdr/ . Small corrections have been applied to match to the actual pupil from internal GMTO files.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 25.448 meters.
    with_spiders : boolean
        If this is False, the spiders will be left out. Default: True.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    Field generator
        The GMT aperture.
    elt_segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    gmt_outer_diameter = 25.448
    segment_size = 8.365 - 0.072
    off_axis_segment_size = 8.365 - 0.015

    # The spider truss to hold the secondary
    spider_1_width = 0.119
    spider_2_width = 0.115
    radius_spider_1 = 2.386
    radius_spider_2 = 2.409
    offset_spider_2 = -0.05
    truss_size = 4.93

    # 0.359 mm from the off-axis segment to the on-axis segment
    segment_gap = 0.359 + 0.088
    off_axis_tilt = np.deg2rad(13.522)
    central_hole_size = 3.495 / segment_size
    segment_distance = segment_size / 2 + off_axis_segment_size / 2 * np.cos(off_axis_tilt) + segment_gap

    if normalized:
        segment_size /= gmt_outer_diameter
        off_axis_segment_size /= gmt_outer_diameter
        segment_gap /= gmt_outer_diameter
        segment_distance /= gmt_outer_diameter
        truss_size /= gmt_outer_diameter
        spider_1_width /= gmt_outer_diameter
        spider_2_width /= gmt_outer_diameter
        radius_spider_1 /= gmt_outer_diameter
        radius_spider_2 /= gmt_outer_diameter
        offset_spider_2 /= gmt_outer_diameter

    def make_diverging_spider(position, start_width, divergence, orientation):
        def func(grid):
            y = grid.shifted(position).rotated(orientation).y
            x = grid.shifted(position).rotated(orientation).x
            return Field(abs(y) < (np.sin(divergence) * abs(x) + start_width / 2) * (x >= 0), grid)
        return func

    def make_central_gmt_segment(grid):
        center_segment = make_obstructed_circular_aperture(segment_size, central_hole_size)(grid)
        if with_spiders:
            spider_attachement_mask = 1 - make_regular_polygon_aperture(3, truss_size, angle=np.pi, center=None)(grid)

            spider_mask = grid.ones()
            for i in range(3):
                offset_angle = 2 * np.pi / 3 * i
                spider_1_start = radius_spider_1 * np.array([np.sin(offset_angle), np.cos(offset_angle)])
                spider_mask *= 1 - make_diverging_spider(spider_1_start, spider_1_width, np.deg2rad(1.16), np.deg2rad(9.83) + offset_angle)(grid)
                spider_mask *= 1 - make_diverging_spider(spider_1_start, spider_1_width, np.deg2rad(1.16), np.deg2rad(180.0 - 9.83) + offset_angle)(grid)

                spider_2_start = radius_spider_2 * np.array([np.sin(offset_angle), np.cos(offset_angle)]) + offset_spider_2 * np.array([np.cos(offset_angle), -np.sin(offset_angle)])
                spider_mask *= 1 - make_diverging_spider(spider_2_start, spider_2_width, np.deg2rad(0.0), np.deg2rad(-11.2) + offset_angle)(grid)

            return center_segment * spider_mask * spider_attachement_mask
        else:
            return center_segment

    segment_functions = [make_central_gmt_segment]
    for i in range(6):
        rotation_angle = np.pi / 3 * i
        xc = segment_distance * np.cos(rotation_angle)
        yc = segment_distance * np.sin(rotation_angle)

        aperture = make_elliptical_aperture([off_axis_segment_size * np.cos(off_axis_tilt), off_axis_segment_size], center=[xc, yc], angle=-rotation_angle)
        segment_functions.append(aperture)

    # Center segment obscurations
    def make_aperture(grid):
        aperture = grid.zeros()
        for segment in segment_functions:
            aperture += segment(grid)
        return aperture

    if return_segments:
        return make_aperture, segment_functions
    else:
        return make_aperture

def make_tmt_aperture(normalized=False, with_spiders=True, segment_transmissions=1, return_segments=False):
    '''Make the Thirty-Meter Telescope aperture.

    The aperture is based on the description from https://www.tmt.org/page/optics. The size of
    the secondary and the spiders were derived from Figure 5 of [JensenClem2021]_

    .. [JensenClem2021] Rebecca Jensen-Clem et al. "The Planetary Systems Imager Adaptive Optics System:
        An Initial Optical Design and Performance Analysis Tools for the PSI-Red AO System" SPIE Vol. 11823 (2021)

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 30.0 meters.
    with_spiders : boolean
        If this is False, the spiders will be left out. Default: True.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission
        will be used for all segments.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    Field generator
        The TMT aperture.
    tmt_segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    tmt_outer_diameter = 30.0     # meter
    spider_width = 0.22     # meter
    segment_size = 1.44     # meter
    segment_gap = 0.0025     # meter
    inner_diameter = 2.5 * segment_size
    central_obscuration = 3.636     # meter

    if normalized:
        spider_width /= tmt_outer_diameter
        segment_size /= tmt_outer_diameter
        segment_gap /= tmt_outer_diameter
        inner_diameter /= tmt_outer_diameter
        central_obscuration /= tmt_outer_diameter
        tmt_outer_diameter = 1.0

    segment_positions = make_hexagonal_grid(segment_size * np.sqrt(3) / 2 + segment_gap, 13, pointy_top=False)

    # remove the first ring and the central segment
    missing_segments_mask = (1 - make_hexagonal_aperture(inner_diameter * 2 / np.sqrt(3))(segment_positions)) > 0
    segment_positions = segment_positions.subset(missing_segments_mask)

    # Clip the outer segments
    inscribed_circle = make_circular_aperture(0.98 * tmt_outer_diameter)(segment_positions) > 0
    segment_positions = segment_positions.subset(inscribed_circle)

    segment_shape = make_hexagonal_aperture(segment_size, angle=np.pi / 2)

    if return_segments:
        tmt_aperture_function, tmt_segments = make_segmented_aperture(segment_shape, segment_positions, segment_transmissions, return_segments=return_segments)
    else:
        tmt_aperture_function = make_segmented_aperture(segment_shape, segment_positions, segment_transmissions)

    spiders = [make_spider_infinite([0, 0], 60 * i + 30, spider_width) for i in range(6)]

    def tmt_aperture_with_spiders(grid):
        aperture = tmt_aperture_function(grid) * (1 - make_circular_aperture(central_obscuration)(grid))

        if with_spiders:
            for spider in spiders:
                aperture *= spider(grid)

        return aperture

    if with_spiders and return_segments:
        # Use function to return the lambda, to avoid incorrect binding of variables
        def spider_func(grid):
            spider_aperture = grid.ones()
            for spider in spiders:
                spider_aperture *= spider(grid)
            return spider_aperture

        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider_func(grid) * (1 - make_circular_aperture(central_obscuration)(grid))

        tmt_segments = [segment_with_spider(s) for s in tmt_segments]
    elif not with_spiders and return_segments:
        def segment_with_central_obscuration(segment):
            return lambda grid: segment(grid) * (1 - make_circular_aperture(central_obscuration)(grid))

        tmt_segments = [segment_with_central_obscuration(s) for s in tmt_segments]

    if return_segments:
        return tmt_aperture_with_spiders, tmt_segments
    else:
        return tmt_aperture_with_spiders

def make_habex_aperture(normalized=False):
    '''Make the HabEx aperture.

    This aperture is based on the HabEx final report 2019. As the HabEx is unobscured,
    there is no option to exclude spiders.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 4.0 meters.

    Returns
    -------
    Field generator
        The HabEx telescope aperture.
    '''
    pupil_diameter = 4.0  # meter

    if normalized:
        pupil_diameter = 1

    return make_circular_aperture(pupil_diameter)

def make_hst_aperture(normalized=False, with_spiders=True, with_pads=True):
    '''Make the Hubble Space Telescope aperture.

    This function uses values from TinyTim [Krist2011]_.

    .. [Krist2011] Krist, Hook and Stoehr, "20 years of Hubble Space Telescope optical modelling using Tiny Tim" Proc. SPIE 8127 (2011).

    .. note::
        This aperture only includes the primary and secondary masks, not any masks
        internal to the instruments.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 2.4 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Hubble Space Telescope aperture.
    '''
    pupil_diameter = 2.4  # meter

    # All these values are relative to the pupil diameter.
    secondary_obscuration_ratio = 0.330
    spider_width = 0.022 / 2

    pad_v3 = np.array([0.8921, -0.4615, -0.4564]) / 2
    pad_v2 = np.array([0.0000, 0.7555, -0.7606]) / 2
    pad_radii = np.array([0.065, 0.065, 0.065]) / 2

    if normalized:
        pupil_diameter = 1
    else:
        spider_width *= pupil_diameter

        pad_v3 *= pupil_diameter
        pad_v2 *= pupil_diameter
        pad_radii *= pupil_diameter

    num_spiders = 4 if with_spiders else 0

    ota = make_obstructed_circular_aperture(pupil_diameter, secondary_obscuration_ratio, num_spiders, spider_width)

    if not with_pads:
        return ota

    pads = [make_obstruction(make_circular_aperture(2 * r, [-v2, v3])) for v3, v2, r in zip(pad_v3, pad_v2, pad_radii)]

    def func(grid):
        return ota(grid) * pads[0](grid) * pads[1](grid) * pads[2](grid)

    return func

def make_jwst_aperture(normalized=False, with_spiders=True, return_segments=False):
    '''Make James Webb Space Telescope (JWST) aperture.

    This function uses coordinates from [WebbPSF]_ v1.0.0. WebbPSF itself got the
    coordinate values from an internal spreadsheet by Paul Lightsey "2010.03.16
    Transmission X Area Budget.xls", which was based in turn on Ball Aerospace
    drawing # 2220169 Rev B and the OTE Cryogenic Optics Interface Control Document,
    Ball Aerospace doc # C327693.

    .. [WebbPSF] The James Webb Space Telescope PSF simulation tool.
        (https://github.com/spacetelescope/webbpsf)

    .. note::
        This function uses the pre-launch aperture for JWST and does not include
        any optical difference maps, filter transmission curves or any internal
        masks in any of the instruments. Use WebbPSF for more complete simulations.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 6.603464 meters.
    with_spiders : boolean
        If this is False, the spiders will be left out. Default: True.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    aperture : Field generator
        The JWST aperture.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    '''
    pupil_diameter = 6.603464  # meter

    segment_parameters = {
        'A1-1': np.array([
            [-0.38101, 0.667604],
            [-0.758826, 1.321999],
            [-0.38101, 1.976407],
            [0.38101, 1.976407],
            [0.758826, 1.321999],
            [0.38101, 0.667604]]),
        'A2-2': np.array([
            [0.38765702, 0.66376634],
            [0.76547172, 1.31816209],
            [1.52111367, 1.31816784],
            [1.90212367, 0.65823916],
            [1.52429772, 0.00383691],
            [0.76866702, 0.00383766]]),
        'A3-3': np.array([
            [0.76866702, -0.00383766],
            [1.52429772, -0.00383691],
            [1.90212367, -0.65823916],
            [1.52111367, -1.31816784],
            [0.76547172, -1.31816209],
            [0.38765702, -0.66376634]]),
        'A4-4': np.array([
            [0.38101, -0.667604],
            [0.758826, -1.321999],
            [0.38101, -1.976407],
            [-0.38101, -1.976407],
            [-0.758826, -1.321999],
            [-0.38101, -0.667604]]),
        'A5-5': np.array([
            [-0.38765702, -0.66376634],
            [-0.76547172, -1.31816209],
            [-1.52111367, -1.31816784],
            [-1.90212367, -0.65823916],
            [-1.52429772, -0.00383691],
            [-0.76866702, -0.00383766]]),
        'A6-6': np.array([
            [-0.76866702, 0.00383766],
            [-1.52429772, 0.00383691],
            [-1.90212367, 0.65823916],
            [-1.52111367, 1.31816784],
            [-0.76547172, 1.31816209],
            [-0.38765702, 0.66376634]]),
        'B1-7': np.array([
            [0.38101, 3.279674],
            [0.758826, 2.631791],
            [0.38101, 1.98402],
            [-0.38101, 1.98402],
            [-0.758826, 2.631791],
            [-0.38101, 3.279674]]),
        'B2-9': np.array([
            [3.030786, 1.30987266],
            [2.65861086, 0.65873291],
            [1.90871672, 0.66204566],
            [1.52770672, 1.32197434],
            [1.89978486, 1.97305809],
            [2.649776, 1.96980134]]),
        'B3-11': np.array([
            [2.649776, -1.96980134],
            [1.89978486, -1.97305809],
            [1.52770672, -1.32197434],
            [1.90871672, -0.66204566],
            [2.65861086, -0.65873291],
            [3.030786, -1.30987266]]),
        'B4-13': np.array([
            [-0.38101, -3.279674],
            [-0.758826, -2.631791],
            [-0.38101, -1.98402],
            [0.38101, -1.98402],
            [0.758826, -2.631791],
            [0.38101, -3.279674]]),
        'B5-15': np.array([
            [-3.030786, -1.30987266],
            [-2.65861086, -0.65873291],
            [-1.90871672, -0.66204566],
            [-1.52770672, -1.32197434],
            [-1.89978486, -1.97305809],
            [-2.649776, -1.96980134]]),
        'B6-17': np.array([
            [-2.649776, 1.96980134],
            [-1.89978486, 1.97305809],
            [-1.52770672, 1.32197434],
            [-1.90871672, 0.66204566],
            [-2.65861086, 0.65873291],
            [-3.030786, 1.30987266]]),
        'C1-8': np.array([
            [0.765201, 2.627516],
            [1.517956, 2.629178],
            [1.892896, 1.976441],
            [1.521076, 1.325812],
            [0.765454, 1.325807],
            [0.387649, 1.980196]]),
        'C2-10': np.array([
            [2.6580961, 0.651074495],
            [3.03591294, 5.42172989e-07],
            [2.65809612, -0.651075523],
            [1.90872487, -0.654384457],
            [1.53090954, 8.90571587e-07],
            [1.90872454, .654384118]]),
        'C3-12': np.array([
            [1.8928951, -1.97644151],
            [1.51795694, -2.62917746],
            [0.76520012, -2.62751652],
            [0.38764887, -1.98019646],
            [0.76545554, -1.32580611],
            [1.52107554, -1.32581188]]),
        'C4-14': np.array([
            [-0.765201, -2.627516],
            [-1.517956, -2.629178],
            [-1.892896, -1.976441],
            [-1.521076, -1.325812],
            [-0.765454, -1.325807],
            [-0.387649, -1.980196]]),
        'C5-16': np.array([
            [-2.6580961, -.651074495],
            [-3.03591294, -5.42172990e-07],
            [-2.65809612, .651075523],
            [-1.90872487, .654384457],
            [-1.53090954, -8.90571587e-07],
            [-1.90872454, -.654384118]]),
        'C6-18': np.array([
            [-1.8928951, 1.97644151],
            [-1.51795694, 2.62917746],
            [-0.76520012, 2.62751652],
            [-0.38764887, 1.98019646],
            [-0.76545554, 1.32580611],
            [-1.52107554, 1.32581188]]),
    }

    strut_parameters = {
        "strut1": np.array([
            [-0.05301375, -0.0306075],
            [1.59698625, -2.88849133],
            [1.70301375, -2.82727633],
            [0.05301375, 0.0306075],
            [-0.05301375, -0.0306075]]),
        "strut2": np.array([
            [-0.05301375, 0.0306075],
            [-1.70301375, -2.82727633],
            [-1.59698625, -2.88849133],
            [0.05301375, -0.0306075],
            [-0.05301375, 0.0306075]]),
        "strut3": np.array([
            [5.94350000e-02, -1.45573765e-17],
            [5.94350000e-02, 3.30000000e+00],
            [-5.94350000e-02, 3.30000000e+00],
            [-5.94350000e-02, 1.45573765e-17],
            [5.94350000e-02, -1.45573765e-17]]),
        "strut3_bumps": np.array([
            [0.059435, 0.666],
            [0.059435, 2.14627],
            [0.082595, 2.14627],
            [0.082595, 2.3645],
            [0.059435, 2.3645],
            [0.059435, 2.48335],
            [0.069795, 2.48335],
            [0.069795, 2.54445],
            [0.059435, 2.54445],
            [0.059435, 3.279674],
            [-0.059435, 3.279674],
            [-0.059435, 2.54445],
            [-0.069795, 2.54445],
            [-0.069795, 2.48335],
            [-0.059435, 2.48335],
            [-0.059435, 2.3645],
            [-0.082595, 2.3645],
            [-0.082595, 2.14627],
            [-0.059435, 2.14627],
            [-0.059435, 0.666]])
    }

    if normalized:
        for val in segment_parameters.values():
            val /= pupil_diameter

        for val in strut_parameters.values():
            val /= pupil_diameter

        pupil_diameter = 1

    segments = [make_irregular_polygon_aperture(vertices) for vertices in segment_parameters.values()]
    struts = [make_obstruction(make_irregular_polygon_aperture(vertices)) for vertices in strut_parameters.values()]

    def func(grid):
        res = sum((segment(grid) for segment in segments))

        if with_spiders:
            for strut in struts:
                res *= strut(grid)

        return res

    if not return_segments:
        return func

    if with_spiders:
        def segment_with_strut(segment):
            def func(grid):
                res = segment(grid)

                for strut in struts:
                    res *= strut(grid)

                return res
            return func

        segments = [segment_with_strut(segment) for segment in segments]

    return func, segments

def make_keck_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, gap_padding=10, segment_transmissions=1, return_segments=False):
    """Make the Keck aperture.

    This code creates a Keck-like aperture matching values used in [vanKooten2022a]_ and [vanKooten2022b]_ as well as
    being verified by Keck personnel to match internal simulation efforts.

    .. [vanKooten2022a] Maaike van Kooten et al., "Predictive wavefront control on Keck II adaptive optics bench: on-sky coronagraphic results." JATIS 8 (2022): 029006
    .. [vanKooten2022b] Maaike van Kooten et al., "On-sky Reconstruction of Keck Primary Mirror Piston Offsets Using a Zernike Wavefront Sensor." The Astrophysical Journal 932 (2022): 2, 109.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 10.95 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    gap_padding : scalar
        Arbitrary padding of gap size to represent gaps on smaller arrays - this effectively
        makes the gaps larger and the segments smaller to preserve the same segment pitch.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission
        will be used for all segments.
    return_segments : boolean
        If this is True, the segments will also be returned as a list of Field generators.

    Returns
    -------
    aperture : Field generator
        The Keck aperture.
    segments : list of Field generators
        The segments. Only returned when `return_segments` is True.
    """
    pupil_diameter = 10.95  # m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3) / 2 * 1.8  # m actual segment flat-to-flat diameter
    central_obscuration_diameter = 2.6  # m
    actual_segment_gap = 0.003  # m actual gap size between segments
    spider_width = 2.6e-2  # m actual strut size
    num_rings = 3  # number of full rings of hexagons around central segment

    if normalized:
        actual_segment_flat_diameter /= pupil_diameter
        actual_segment_gap /= pupil_diameter
        spider_width /= pupil_diameter
        central_obscuration_diameter /= pupil_diameter
        pupil_diameter = 1.0

    # padding out the segmentation gaps so they are visible and not sub-pixel
    segment_gap = actual_segment_gap * gap_padding
    if not with_segment_gaps:
        segment_gap = 0

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)

    segment = make_hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    if with_spiders:
        spider1 = make_spider_infinite([0, 0], 0, spider_width)
        spider2 = make_spider_infinite([0, 0], 60, spider_width)
        spider3 = make_spider_infinite([0, 0], 120, spider_width)
        spider4 = make_spider_infinite([0, 0], 180, spider_width)
        spider5 = make_spider_infinite([0, 0], 240, spider_width)
        spider6 = make_spider_infinite([0, 0], 300, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=return_segments)
    if return_segments:
        segmented_aperture, segments = segmented_aperture

    def func(grid):
        res = segmented_aperture(grid) * (1 - make_circular_aperture(central_obscuration_diameter)(grid))

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * spider5(grid) * spider6(grid)

        return Field(res, grid)

    if with_spiders and return_segments:
        # Use function to return the lambda, to avoid incorrect binding of variables
        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * spider5(grid) * spider6(grid)

        segments = [segment_with_spider(s) for s in segments]

    if return_segments:
        def segment_with_central_obscuration(segment):
            return lambda grid: segment(grid) * (1 - make_circular_aperture(central_obscuration_diameter)(grid))

        segments = [segment_with_central_obscuration(segment) for segment in segments]

        return func, segments
    else:
        return func
