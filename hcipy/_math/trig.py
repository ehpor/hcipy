import math

def deg2rad(degrees):
    """Convert degrees to radians.

    Parameters
    ----------
    degrees : array-like
        Angle in degrees.

    Returns
    -------
    array-like
        Angle in radians.
    """
    return degrees * (math.pi / 180.0)

def rad2deg(radians):
    """Convert radians to degrees.

    Parameters
    ----------
    radians : array-like
        Angle in radians.

    Returns
    -------
    array-like
        Angle in degrees.
    """
    return radians * (180.0 / math.pi)
