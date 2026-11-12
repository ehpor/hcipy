import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import binary_dilation, label

from .mode_basis import ModeBasis
from ..field import Field

def make_kirchhoff_love_basis(bending_stiffness, num_modes, fixed_points=None,
                              poisson_ratio=0, mass_density=None, return_frequencies=False):
    '''Make the mode basis of a flat membrane with Kirchhoff-Love bending.

    The membrane is modelled as a Kirchhoff-Love plate with a possibly
    varying bending stiffness. The membrane is free (not clamped) at its
    edge; the free (natural) boundary conditions emerge from a variational
    discretization of the bending energy. Points of the membrane can be
    clamped to an external surface through the `fixed_points` parameter.

    The modes are calculated by discretizing the bending energy and the mass
    of the membrane on the grid, and solving the generalized eigenvalue
    problem

    .. math:: K u = \\omega^2 M u,

    where `K` and `M` are the discretized bending stiffness and mass matrices,
    and :math:`\\omega = 2\\pi f` is the angular mode frequency. The modes are
    sorted by ascending mode frequency :math:`f`.

    Parameters
    ----------
    bending_stiffness : Field
        The bending stiffness :math:`D(x)` of the membrane, on a
        two-dimensional regular Cartesian grid. Values should be non-negative.
        The membrane is formed by the points where the value is non-zero. To
        reproduce a free edge, the active region should be inset from the edge
        of the grid. In SI units the bending stiffness is in N m (Pa m^3).
    num_modes : int
        The number of modes to calculate.
    fixed_points : Field or None
        A Field on the same grid as `bending_stiffness`, marking the points
        (with values above 0.5) at which the membrane is clamped to an
        external surface. The modes are set to zero at these points. These
        points don't need to be at the edge of the membrane.
    poisson_ratio : scalar
        The Poisson ratio of the material.
    mass_density : Field or None
        A Field on the same grid as `bending_stiffness`, giving the mass per
        unit area of the membrane in kg / m^2. If this is None, a uniform mass
        density of one is assumed.
    return_frequencies : boolean
        If True, the mode frequencies are returned alongside the mode basis.

    Returns
    -------
    modes : ModeBasis or tuple
        The mode basis of the membrane.
    frequencies : ndarray
        The mode frequencies :math:`f` in Hz. These are only returned when
        `return_frequencies` is true. The frequencies are in Hz if the bending
        stiffness is in N m, the mass density in kg / m^2, and the grid
        coordinates in meters.

    Raises
    ------
    ValueError
        If the inputs are not valid.
    '''
    if not isinstance(bending_stiffness, Field):
        raise ValueError('bending_stiffness should be a Field.')
    grid = bending_stiffness.grid

    if not grid.is_('cartesian') or not grid.is_regular:
        raise ValueError('The grid of bending_stiffness should be a regular Cartesian grid.')

    delta, dims, _ = grid.regular_coords

    if len(dims) != 2:
        raise ValueError('The grid of bending_stiffness should be two-dimensional.')

    Nx = int(round(dims[0]))
    Ny = int(round(dims[1]))
    dx = float(np.asarray(delta)[0])
    dy = float(np.asarray(delta)[1])

    if dx <= 0 or dy <= 0:
        raise ValueError('The grid of bending_stiffness should have positive spacing.')

    size_grid = Nx * Ny

    stiffness = np.asarray(bending_stiffness, dtype=float).reshape(-1)
    if stiffness.size != size_grid:
        raise ValueError('bending_stiffness should be defined on the grid.')
    stiffness = stiffness.reshape((Ny, Nx))

    fixed = None
    if fixed_points is not None:
        if not isinstance(fixed_points, Field):
            raise ValueError('fixed_points should be a Field.')
        fp = np.asarray(fixed_points, dtype=float).reshape(-1)
        if fp.size != size_grid:
            raise ValueError('fixed_points should be defined on the same grid as bending_stiffness.')
        fixed = fp.reshape((Ny, Nx)) > 0.5

    if mass_density is not None:
        if not isinstance(mass_density, Field):
            raise ValueError('mass_density should be a Field.')
        md = np.asarray(mass_density, dtype=float).reshape(-1)
        if md.size != size_grid:
            raise ValueError('mass_density should be defined on the same grid as bending_stiffness.')
        mass = md.reshape((Ny, Nx))
    else:
        mass = np.ones((Ny, Nx))

    active = stiffness > 0

    if not np.any(active):
        raise ValueError('bending_stiffness should be non-zero at at least one point.')
    if np.any(stiffness < 0):
        raise ValueError('bending_stiffness should be non-negative.')
    if np.any(mass[active] < 0):
        raise ValueError('mass_density should be non-negative on the membrane.')

    # The degrees of freedom are the active points of the membrane, plus a
    # one-pixel halo around it. The halo allows the free (natural) boundary
    # conditions to emerge from the variational principle.
    halo = binary_dilation(active, structure=np.ones((3, 3))) & ~active

    dof = active | halo
    if fixed is not None:
        dof[fixed] = False

    node_index = np.full((Ny, Nx), -1, dtype=int)
    node_index[dof] = np.arange(dof.sum())
    num_dof = int(dof.sum())

    if num_modes >= num_dof:
        raise ValueError('num_modes should be smaller than the number of degrees of freedom.')

    ya, xa = np.nonzero(active)
    n_act = ya.size

    def neighbor(dyi, dxi):
        iy = ya + dyi
        ix = xa + dxi
        inside = (iy >= 0) & (iy < Ny) & (ix >= 0) & (ix < Nx)

        res = np.full(n_act, -1, dtype=int)
        res[inside] = node_index[np.clip(iy, 0, Ny - 1)[inside], np.clip(ix, 0, Nx - 1)[inside]]
        return res

    center = node_index[ya, xa]
    right = neighbor(0, 1)
    left = neighbor(0, -1)
    down = neighbor(1, 0)
    up = neighbor(-1, 0)

    corr_pp = neighbor(1, 1)
    corr_pm = neighbor(1, -1)
    corr_mp = neighbor(-1, 1)
    corr_mm = neighbor(-1, -1)

    def make_stencil(indices, values):
        '''Zero out the invalid (out of grid / not a degree of freedom) entries.'''
        bad = indices < 0
        values = values.copy()
        values[bad] = 0.0
        indices = np.maximum(indices, 0)
        return indices, values

    sxx_i = np.stack([right, center, left], axis=-1)
    sxx_v = np.zeros_like(sxx_i, dtype=float)
    sxx_v[:] = 1.0 / dx**2
    sxx_v[:, 1] = -2.0 / dx**2
    sxx_i, sxx_v = make_stencil(sxx_i, sxx_v)

    syy_i = np.stack([down, center, up], axis=-1)
    syy_v = np.zeros_like(syy_i, dtype=float)
    syy_v[:] = 1.0 / dy**2
    syy_v[:, 1] = -2.0 / dy**2
    syy_i, syy_v = make_stencil(syy_i, syy_v)

    sxy_i = np.stack([corr_pp, corr_mm, corr_pm, corr_mp], axis=-1)
    sxy_v = np.zeros_like(sxy_i, dtype=float)
    sxy_v[:, 0] = 1.0 / (4 * dx * dy)
    sxy_v[:, 1] = 1.0 / (4 * dx * dy)
    sxy_v[:, 2] = -1.0 / (4 * dx * dy)
    sxy_v[:, 3] = -1.0 / (4 * dx * dy)
    sxy_i, sxy_v = make_stencil(sxy_i, sxy_v)

    def outer_prods(idx_a, val_a):
        n, k = idx_a.shape
        rows = np.broadcast_to(idx_a[:, :, None], (n, k, k)).ravel()
        cols = np.broadcast_to(idx_a[:, None, :], (n, k, k)).ravel()
        vals = val_a[:, :, None] * val_a[:, None, :]
        return rows, cols, vals

    def cross_prods(idx_a, val_a, idx_b, val_b):
        n = idx_a.shape[0]
        ka = idx_a.shape[1]
        kb = idx_b.shape[1]
        rows = np.broadcast_to(idx_a[:, :, None], (n, ka, kb)).ravel()
        cols = np.broadcast_to(idx_b[:, None, :], (n, ka, kb)).ravel()
        vals = val_a[:, :, None] * val_b[:, None, :]
        return rows, cols, vals

    stiffness_scale = stiffness[ya, xa] * (dx * dy)

    rows = []
    cols = []
    vals = []

    # u_xx^2 and u_yy^2 terms.
    r, c, v = outer_prods(sxx_i, sxx_v)
    rows.append(r); cols.append(c); vals.append((stiffness_scale[:, None, None] * v).ravel())

    r, c, v = outer_prods(syy_i, syy_v)
    rows.append(r); cols.append(c); vals.append((stiffness_scale[:, None, None] * v).ravel())

    # 2 (1 - nu) u_xy^2 term.
    r, c, v = outer_prods(sxy_i, sxy_v)
    rows.append(r); cols.append(c); vals.append((2 * (1 - poisson_ratio) * stiffness_scale[:, None, None] * v).ravel())

    # 2 nu u_xx u_yy coupling terms.
    r, c, v = cross_prods(sxx_i, sxx_v, syy_i, syy_v)
    rows.append(r); cols.append(c); vals.append((poisson_ratio * stiffness_scale[:, None, None] * v).ravel())

    r, c, v = cross_prods(syy_i, syy_v, sxx_i, sxx_v)
    rows.append(r); cols.append(c); vals.append((poisson_ratio * stiffness_scale[:, None, None] * v).ravel())

    rows_full = np.concatenate(rows)
    cols_full = np.concatenate(cols)
    vals_full = np.concatenate(vals)

    K = scipy.sparse.csr_matrix((vals_full, (rows_full, cols_full)), shape=(num_dof, num_dof))

    mass_mat = np.zeros((Ny, Nx))
    mass_mat[active] = mass[active] * (dx * dy)

    mass_scale = np.max(mass_mat[active])

    if mass_scale == 0:
        raise ValueError('mass_density should be non-zero at at least one point of the membrane.')

    mass_mat[active] = np.maximum(mass_mat[active], mass_scale * 1e-12)
    mass_mat[halo] = mass_scale * 1e-9

    M = scipy.sparse.diags(np.maximum(mass_mat[dof], 1e-30))

    num_components = label(active, structure=np.ones((3, 3)))[1]
    extra = 3 * num_components + 3
    k_arpack = min(num_modes + extra, num_dof - 1)

    vals_eig, vecs = scipy.sparse.linalg.eigsh(K, k=k_arpack, M=M, sigma=-1.0, which='LM')

    order = np.argsort(vals_eig)
    vals_eig = vals_eig[order]
    vecs = vecs[:, order]

    # Remove the rigid-body modes, which have a frequency of zero.
    threshold = max(np.max(vals_eig) * 1e-8, np.finfo(float).eps)
    keep = vals_eig > threshold

    if keep.sum() < num_modes:
        raise ValueError('Not enough modes with a non-zero frequency could be found. Consider increasing the grid resolution, or decreasing num_modes.')

    mode_freqs = vals_eig[keep][:num_modes]
    mode_vecs = vecs[:, keep][:, :num_modes]

    # Normalize the modes w.r.t. the mass matrix.
    m_diag = np.asarray(M.diagonal())
    mode_vecs = mode_vecs / np.sqrt(np.sum(mode_vecs**2 * m_diag[:, None], axis=0))[None, :]

    modes_full = np.zeros((size_grid, num_modes))
    modes_full[np.flatnonzero(dof)] = mode_vecs

    modes = [Field(modes_full[:, i], grid) for i in range(num_modes)]
    mode_basis = ModeBasis(modes, grid)

    if return_frequencies:
        return mode_basis, np.sqrt(mode_freqs) / (2 * np.pi)
    else:
        return mode_basis
