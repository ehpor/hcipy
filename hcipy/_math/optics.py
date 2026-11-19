import numexpr as ne
from array_api_compat import is_numpy_array

from .backends import array_namespace

def _tensor_components(electric_field):
    return electric_field[0, 0, :], electric_field[0, 1, :], electric_field[1, 0, :], electric_field[1, 1, :]

def _require_stokes_vector(input_stokes_vector):
    if input_stokes_vector is None:
        raise ValueError('An input Stokes vector is required for a tensor electric field.')

    return input_stokes_vector

def _tensor_stokes_numpy(electric_field, input_stokes_vector, expressions):
    x, y, z, w = _tensor_components(electric_field)
    a, b, c, d = input_stokes_vector

    m11, m12, m13, m14 = expressions
    expression = '0.5 * ((' + m11 + ') * a + (' + m12 + ') * b + (' + m13 + ') * c +  (' + m14 + ') * d)'

    return ne.evaluate(expression, local_dict={'x': x, 'y': y, 'z': z, 'w': w, 'a': a, 'b': b, 'c': c, 'd': d})

def stokes_i_numpy(electric_field, input_stokes_vector):
    if electric_field.ndim == 1:
        result = ne.evaluate('real(abs(elec))**2', local_dict={'elec': electric_field})

        return result

    if electric_field.ndim == 2:
        return stokes_i_fallback(electric_field, input_stokes_vector)

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        result = _tensor_stokes_numpy(electric_field, input_stokes_vector, (
            'real(abs(x))**2 + real(abs(y))**2 + real(abs(z))**2 + real(abs(w))**2',
            'real(abs(x))**2 - real(abs(y))**2 + real(abs(z))**2 - real(abs(w))**2',
            '2 * (real(x) * real(y) + imag(x) * imag(y) + real(z) * real(w) + imag(z) * imag(w))',
            '2 * (-real(x) * imag(y) + imag(x) * real(y) - real(z) * imag(w) + imag(z) * real(w))',
        ))

        return result

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_i_fallback(electric_field, input_stokes_vector):
    xp = array_namespace(electric_field)

    if electric_field.ndim == 1:
        return xp.abs(electric_field)**2

    if electric_field.ndim == 2:
        return xp.sum(xp.abs(electric_field)**2, axis=0)

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        x, y, z, w = _tensor_components(electric_field)
        a, b, c, d = input_stokes_vector

        m11 = xp.abs(x)**2 + xp.abs(y)**2 + xp.abs(z)**2 + xp.abs(w)**2
        m12 = xp.abs(x)**2 - xp.abs(y)**2 + xp.abs(z)**2 - xp.abs(w)**2
        m13 = 2 * (xp.real(x) * xp.real(y) + xp.imag(x) * xp.imag(y) + xp.real(z) * xp.real(w) + xp.imag(z) * xp.imag(w))
        m14 = 2 * (-xp.real(x) * xp.imag(y) + xp.imag(x) * xp.real(y) - xp.real(z) * xp.imag(w) + xp.imag(z) * xp.real(w))

        return 0.5 * (m11 * a + m12 * b + m13 * c + m14 * d)

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_i(electric_field, input_stokes_vector=None):
    '''Calculate the Stokes I component of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.
    input_stokes_vector : Array or None
        The input Stokes vector. This is required for a 2-tensor electric
        field.

    Returns
    -------
    Array
        The Stokes I component.
    '''
    if is_numpy_array(electric_field):
        return stokes_i_numpy(electric_field, input_stokes_vector)

    return stokes_i_fallback(electric_field, input_stokes_vector)

def stokes_q_numpy(electric_field, input_stokes_vector):
    if electric_field.ndim == 1:
        return stokes_q_fallback(electric_field, input_stokes_vector)

    if electric_field.ndim == 2:
        result = ne.evaluate(
            'real(abs(a))**2 - real(abs(b))**2',
            local_dict={'a': electric_field[0, :], 'b': electric_field[1, :]},
        )

        return result

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        result = _tensor_stokes_numpy(electric_field, input_stokes_vector, (
            'real(abs(x))**2 + real(abs(y))**2 - real(abs(z))**2 - real(abs(w))**2',
            'real(abs(x))**2 - real(abs(y))**2 - real(abs(z))**2 + real(abs(w))**2',
            '2 * (real(x) * real(y) + imag(x) * imag(y) - real(z) * real(w) - imag(z) * imag(w))',
            '2 * (-real(x) * imag(y) + imag(x) * real(y) + real(z) * imag(w) - imag(z) * real(w))',
        ))

        return result

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_q_fallback(electric_field, input_stokes_vector):
    xp = array_namespace(electric_field)

    if electric_field.ndim == 1:
        return xp.zeros_like(xp.real(electric_field))

    if electric_field.ndim == 2:
        return xp.abs(electric_field[0, :])**2 - xp.abs(electric_field[1, :])**2

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        x, y, z, w = _tensor_components(electric_field)
        a, b, c, d = input_stokes_vector

        m21 = xp.abs(x)**2 + xp.abs(y)**2 - xp.abs(z)**2 - xp.abs(w)**2
        m22 = xp.abs(x)**2 - xp.abs(y)**2 - xp.abs(z)**2 + xp.abs(w)**2
        m23 = 2 * (xp.real(x) * xp.real(y) + xp.imag(x) * xp.imag(y) - xp.real(z) * xp.real(w) - xp.imag(z) * xp.imag(w))
        m24 = 2 * (-xp.real(x) * xp.imag(y) + xp.imag(x) * xp.real(y) + xp.real(z) * xp.imag(w) - xp.imag(z) * xp.real(w))

        return 0.5 * (m21 * a + m22 * b + m23 * c + m24 * d)

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_q(electric_field, input_stokes_vector=None):
    '''Calculate the Stokes Q component of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.
    input_stokes_vector : Array or None
        The input Stokes vector. This is required for a 2-tensor electric
        field.

    Returns
    -------
    Array
        The Stokes Q component.
    '''
    if is_numpy_array(electric_field):
        return stokes_q_numpy(electric_field, input_stokes_vector)

    return stokes_q_fallback(electric_field, input_stokes_vector)

def stokes_u_numpy(electric_field, input_stokes_vector):
    if electric_field.ndim == 1:
        return stokes_u_fallback(electric_field, input_stokes_vector)

    if electric_field.ndim == 2:
        result = ne.evaluate(
            '2 * real(a * conj(b))',
            local_dict={'a': electric_field[0, :], 'b': electric_field[1, :]},
        )

        return result

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        result = _tensor_stokes_numpy(electric_field, input_stokes_vector, (
            '2 * (real(x) * real(z) + imag(x) * imag(z) + real(y) * real(w) + imag(y) * imag(w))',
            '2 * (real(x) * real(z) + imag(x) * imag(z) - real(y) * real(w) - imag(y) * imag(w))',
            '2 * (real(x) * real(w) + imag(x) * imag(w) + real(y) * real(z) + imag(y) * imag(z))',
            '2 * (imag(x) * real(w) - real(x) * imag(w) + real(y) * imag(z) - imag(y) * real(z))',
        ))

        return result

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_u_fallback(electric_field, input_stokes_vector):
    xp = array_namespace(electric_field)

    if electric_field.ndim == 1:
        return xp.zeros_like(xp.real(electric_field))

    if electric_field.ndim == 2:
        return 2 * xp.real(electric_field[0, :] * xp.conj(electric_field[1, :]))

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        x, y, z, w = _tensor_components(electric_field)
        a, b, c, d = input_stokes_vector

        m31 = 2 * (xp.real(x) * xp.real(z) + xp.imag(x) * xp.imag(z) + xp.real(y) * xp.real(w) + xp.imag(y) * xp.imag(w))
        m32 = 2 * (xp.real(x) * xp.real(z) + xp.imag(x) * xp.imag(z) - xp.real(y) * xp.real(w) - xp.imag(y) * xp.imag(w))
        m33 = 2 * (xp.real(x) * xp.real(w) + xp.imag(x) * xp.imag(w) + xp.real(y) * xp.real(z) + xp.imag(y) * xp.imag(z))
        m34 = 2 * (xp.imag(x) * xp.real(w) - xp.real(x) * xp.imag(w) + xp.real(y) * xp.imag(z) - xp.imag(y) * xp.real(z))

        return 0.5 * (m31 * a + m32 * b + m33 * c + m34 * d)

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_u(electric_field, input_stokes_vector=None):
    '''Calculate the Stokes U component of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.
    input_stokes_vector : Array or None
        The input Stokes vector. This is required for a 2-tensor electric
        field.

    Returns
    -------
    Array
        The Stokes U component.
    '''
    if is_numpy_array(electric_field):
        return stokes_u_numpy(electric_field, input_stokes_vector)

    return stokes_u_fallback(electric_field, input_stokes_vector)

def stokes_v_numpy(electric_field, input_stokes_vector):
    if electric_field.ndim == 1:
        return stokes_v_fallback(electric_field, input_stokes_vector)

    if electric_field.ndim == 2:
        result = ne.evaluate(
            '-2 * imag(a * conj(b))',
            local_dict={'a': electric_field[0, :], 'b': electric_field[1, :]},
        )

        return result

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        result = _tensor_stokes_numpy(electric_field, input_stokes_vector, (
            '2 * (real(x) * imag(z) - imag(x) * real(z) + real(y) * imag(w) - imag(y) * real(w))',
            '2 * (real(x) * imag(z) - imag(x) * real(z) - real(y) * imag(w) + imag(y) * real(w))',
            '2 * (real(x) * imag(w) - imag(x) * real(w) + real(y) * imag(z) - imag(y) * real(z))',
            '2 * (real(x) * real(w) + imag(x) * imag(w) - real(y) * real(z) - imag(y) * imag(z))',
        ))

        return result

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_v_fallback(electric_field, input_stokes_vector):
    xp = array_namespace(electric_field)

    if electric_field.ndim == 1:
        return xp.zeros_like(xp.real(electric_field))

    if electric_field.ndim == 2:
        return -2 * xp.imag(electric_field[0, :] * xp.conj(electric_field[1, :]))

    if electric_field.ndim == 3:
        input_stokes_vector = _require_stokes_vector(input_stokes_vector)
        x, y, z, w = _tensor_components(electric_field)
        a, b, c, d = input_stokes_vector

        m41 = 2 * (xp.real(x) * xp.imag(z) - xp.imag(x) * xp.real(z) + xp.real(y) * xp.imag(w) - xp.imag(y) * xp.real(w))
        m42 = 2 * (xp.real(x) * xp.imag(z) - xp.imag(x) * xp.real(z) - xp.real(y) * xp.imag(w) + xp.imag(y) * xp.real(w))
        m43 = 2 * (xp.real(x) * xp.imag(w) - xp.imag(x) * xp.real(w) + xp.real(y) * xp.imag(z) - xp.imag(y) * xp.real(z))
        m44 = 2 * (xp.real(x) * xp.real(w) + xp.imag(x) * xp.imag(w) - xp.real(y) * xp.real(z) - xp.imag(y) * xp.imag(z))

        return 0.5 * (m41 * a + m42 * b + m43 * c + m44 * d)

    raise ValueError('The electric field must be a scalar, vector or 2-tensor array.')

def stokes_v(electric_field, input_stokes_vector=None):
    '''Calculate the Stokes V component of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.
    input_stokes_vector : Array or None
        The input Stokes vector. This is required for a 2-tensor electric
        field.

    Returns
    -------
    Array
        The Stokes V component.
    '''
    if is_numpy_array(electric_field):
        return stokes_v_numpy(electric_field, input_stokes_vector)

    return stokes_v_fallback(electric_field, input_stokes_vector)

def polarization_degree_numpy(intensity, q, u, v):
    result = ne.evaluate('sqrt(q**2 + u**2 + v**2) / i', local_dict={'i': intensity, 'q': q, 'u': u, 'v': v})

    return result

def polarization_degree_fallback(intensity, q, u, v):
    xp = array_namespace(intensity, q, u, v)

    return xp.sqrt(q**2 + u**2 + v**2) / intensity

def polarization_degree(intensity, q, u, v):
    '''Calculate the degree of polarization.

    Parameters
    ----------
    intensity : Array
        The Stokes I component.
    q : Array
        The Stokes Q component.
    u : Array
        The Stokes U component.
    v : Array
        The Stokes V component.

    Returns
    -------
    Array
        The degree of polarization.
    '''
    if is_numpy_array(intensity):
        return polarization_degree_numpy(intensity, q, u, v)

    return polarization_degree_fallback(intensity, q, u, v)

def linear_polarization_degree_numpy(intensity, q, u):
    result = ne.evaluate('sqrt(q**2 + u**2) / i', local_dict={'i': intensity, 'q': q, 'u': u})

    return result

def linear_polarization_degree_fallback(intensity, q, u):
    xp = array_namespace(intensity, q, u)

    return xp.sqrt(q**2 + u**2) / intensity

def linear_polarization_degree(intensity, q, u):
    '''Calculate the degree of linear polarization.

    Parameters
    ----------
    intensity : Array
        The Stokes I component.
    q : Array
        The Stokes Q component.
    u : Array
        The Stokes U component.

    Returns
    -------
    Array
        The degree of linear polarization.
    '''
    if is_numpy_array(intensity):
        return linear_polarization_degree_numpy(intensity, q, u)

    return linear_polarization_degree_fallback(intensity, q, u)

def linear_polarization_angle_numpy(q, u):

    result = ne.evaluate('0.5 * arctan2(u, q)', local_dict={'q': q, 'u': u})

    return result

def linear_polarization_angle_fallback(q, u):
    xp = array_namespace(q, u)

    return 0.5 * xp.atan2(u, q)

def linear_polarization_angle(q, u):
    '''Calculate the angle of linear polarization.

    Parameters
    ----------
    q : Array
        The Stokes Q component.
    u : Array
        The Stokes U component.

    Returns
    -------
    Array
        The angle of linear polarization in radians.
    '''
    if is_numpy_array(q):
        return linear_polarization_angle_numpy(q, u)

    return linear_polarization_angle_fallback(q, u)

def ellipticity_numpy(intensity, q, u, v):

    result = ne.evaluate('v / (i + sqrt(q**2 + u**2))', local_dict={'i': intensity, 'q': q, 'u': u, 'v': v})

    return result

def ellipticity_fallback(intensity, q, u, v):
    xp = array_namespace(intensity, q, u, v)

    return v / (intensity + xp.sqrt(q**2 + u**2))

def ellipticity(intensity, q, u, v):
    '''Calculate the ellipticity.

    Parameters
    ----------
    intensity : Array
        The Stokes I component.
    q : Array
        The Stokes Q component.
    u : Array
        The Stokes U component.
    v : Array
        The Stokes V component.

    Returns
    -------
    Array
        The ratio of the minor to major axis of the polarization ellipse.
    '''
    if is_numpy_array(intensity):
        return ellipticity_numpy(intensity, q, u, v)

    return ellipticity_fallback(intensity, q, u, v)

def phase_numpy(electric_field):
    result = ne.evaluate('arctan2(imag(elec), real(elec))', local_dict={'elec': electric_field})

    return result

def phase_fallback(electric_field):
    xp = array_namespace(electric_field)

    return xp.atan2(xp.imag(electric_field), xp.real(electric_field))

def phase(electric_field):
    '''Calculate the phase of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.

    Returns
    -------
    Array
        The phase of the electric field in radians.
    '''
    if is_numpy_array(electric_field):
        return phase_numpy(electric_field)

    return phase_fallback(electric_field)

def power_numpy(electric_field, weights, input_stokes_vector):
    if electric_field.ndim == 1:
        result = ne.evaluate(
            'real(abs(field))**2 * weights',
            local_dict={'field': electric_field, 'weights': weights},
        )

        return result

    return stokes_i_numpy(electric_field, input_stokes_vector) * weights

def power_fallback(electric_field, weights, input_stokes_vector):
    if electric_field.ndim == 1:
        xp = array_namespace(electric_field)

        return xp.abs(electric_field)**2 * weights

    return stokes_i_fallback(electric_field, input_stokes_vector) * weights

def power(electric_field, weights, input_stokes_vector=None):
    '''Calculate the power per pixel of an electric field.

    Parameters
    ----------
    electric_field : Array
        The electric field as a scalar, vector or 2-tensor array.
    weights : Array or scalar
        The weight of each pixel.
    input_stokes_vector : Array or None
        The input Stokes vector. This is required for a 2-tensor electric
        field.

    Returns
    -------
    Array
        The power per pixel.
    '''
    if is_numpy_array(electric_field):
        return power_numpy(electric_field, weights, input_stokes_vector)

    return power_fallback(electric_field, weights, input_stokes_vector)
