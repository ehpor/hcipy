import numpy as np
from hcipy import *
from scipy.special import jv,kn,kv
from matplotlib import pyplot as plt

def jkdiff(m,u,w):
    """Calculate the absolute difference diff = |Jm(u)/Jm+1(u)-Km(w)/Km+1(w)|.

    Can be used to determine the branches of LP modes in a step-index fiber.

    Arguments:

        - m azimuthal number of periods (m=0,1,2,3...)
        - u radial phase constant
        - w radial decay constant

    Returns:

        - diff - Difference

    """
    return np.abs(  jv(m, u)/(u * jv(m+1,u)) - (kn(m,w)/(w*kn(m+1,w))))


def calc_jkdiff_matrix(m, Vmax, pts=300):
    """ calculate the Difference
        diff = |Jm(u)/Jm+1(u)-Km(w)/Km+1(w)|
        for a given m for a matrix
        [0..Vmax] x [0..Vmax] with pts x pts values.

    Arguments:

        - m: azimuthal number of periods (m=0,1,2,3...)
        - Vmax:  maximum V-number, normalized frequency

    Optional Arguments:

        - pts: number of grid points for each of the two
               axes of the matrix

    Returns:

        - jkdiffmatrix
        - uv : u vector (=w vector)
    """
    uv = np.linspace(0, Vmax, pts)
    uu, ww = np.meshgrid(uv, uv)
    uu2 = np.reshape(uu, pts * pts)
    ww2 = np.reshape(ww, pts * pts)
    diff = jkdiff(m, uu2, ww2)
    diff = np.reshape(diff, [pts, pts])
    return diff, uv

def get_intersects(m, V, anglepts=500, peakfindpts=5, maxjkdiff=1e-2):
    """Calculate the intersects of the V-circle with the branches of LPmp for given m

    Arguments:

        - m azimuthal number of periods (m=0,1,2,3...)
        - V  maximum V-number, normalized frequency

    Optional arguments:
        - anglepts: number of points for the circle (default=500)
        - peakfindpts: intersection points are determined by searching
                       for peaks of 1/jkdiff along the V-circle.
                       For an u-w pair to be recognized as peak,
                       it must be a maximum in a surrounding of
                       peakfindpts points.
        - maxjkdiff: sets the maximum value for jkdiff, so that
                     an intersection is still recognized

    Returns:
        - reslist: list of branch intersections found.
            consists of sub-lists [u, w, modename]

    """
    epsi = 1e-5

    angle = np.linspace(np.pi/2.0-epsi, epsi, anglepts)
    w = np.sin(angle) * V
    u = np.cos(angle) * V
    pl = pyfindpeaks(peakfindpts, 1./jkdiff(m, u, w), 1./maxjkdiff)
    res = []
    for ii,p in enumerate(pl):
        res.append([u[p], w[p],"LP%d%d"%(m,ii+1)])
    return res

def pyfindpeaks( environment, valuelist , thresh):
    """Determine peak positions in a list or array of real values.

    Arguments:
      - environment: (INT) a maxima has to be the local maximum in this environment of points
      - valuelist: list or array of points to find the maxima in
      - thresh: a maximum has to be larger than this value

    Returns:
      - listindices: positions of the peaks found

    """
    def limitss(diff,length,pos):
    #this prevents hitting the borders of the array
        mi = np.max( [0, pos-diff])
        ma = np.min( [length, pos+diff])
        return mi,ma
    #range left/right
    half = int( np.floor( environment/2))
    valuelistlength = len(valuelist)
    #pre-filter the peaks above threshold
    abovethresh = np.nonzero( valuelist>thresh )[0]
    i = 0
    peakpos =np.array([],int)
    # circle through the candidates
    while (i < len(abovethresh)):
        mi,ma = limitss(half, valuelistlength, abovethresh[i])
        partialvaluelist = valuelist[mi:ma]
    # is the valuelist value of the actual position the maximum of the environment?
        if valuelist[abovethresh[i]] == max(partialvaluelist):
            peakpos=np.append(peakpos,abovethresh[i])
        i = i+1
    return peakpos

def eigen_value_equation(m, V):
	def cost(u):
		w = np.sqrt(V**2 - u**2)
		diff = jv(m, u)/(u * jv(m+1,u)) - kn(m,w)/(w*kn(m+1,w))
		return abs(diff)
	return cost

def solve_eigenvalue_equation(m, V):
	pass
	

def LP_radial(m, u, w, r):
	"""Calculate the field of a bessel mode LP mode.

	Arguments:
		- m azimuthal number of periods (m=0,1,2,3...)
		- u, w  radial phase constant and radial decay constant
		- x, y transverse coordinates
		- phioff: offset angle, allows to rotate the mode in
					the x-y plane

	Returns:
		- mode: calculated bessel mode

	"""

	# The scaling factor for the continuity condition
	scaling_factor = jv(m,u)/kn(m, w)

	# Find the grid inside and outside the core radius
	mask = r < 1

	# Evaluate the radial mode profile
	mode_field = np.zeros_like(r)
	mode_field[mask] = jv(m, u * r[mask])
	mode_field[~mask] = scaling_factor * kn(m, w * r[~mask])

	return mode_field

def LP_azimuthal(m, theta):
	if m >= 0:
		return np.cos(m * theta)
	else:
		return np.sin(m * theta)

def make_LP_modes(grid, V_number, core_radius, mode_cutoff=None):
	finding_new_modes = True
	m = 0
	num_modes = 0


	# scaled grid
	R, Theta = grid.scaled(1/core_radius).as_('polar').coords
	modes = []
	while finding_new_modes:

		intersects = get_intersects(m, V)
		if len(intersects) > 0:
	
			for mode_solution in intersects:
				
				radial_prodile = LP_radial(m, mode_solution[0], mode_solution[1], R)

				ms = [m,-m] if m > 0 else [m,]
				for mi in ms:
					azimutal_profile = LP_azimuthal(mi, Theta)
					mode_profile = radial_prodile * azimutal_profile
					modes.append(mode_profile)
					num_modes += 1

			m += 1
		else:
			finding_new_modes = False
	
	return ModeBasis(modes, grid)

if __name__ == '__main__':
	um = 1
	
	NA = 0.13
	a = 2.2 * um
	wavelength = 0.656 * um

	# set the V-number
	V = 2 * np.pi / wavelength * a * NA
	grid = make_pupil_grid(128, 10 * a)
	modes = make_LP_modes(grid, V, a)