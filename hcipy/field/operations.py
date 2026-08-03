from .field import Field, NewStyleField, is_field
import numpy as np
import string
from .._math.einsum import einsum
from .._math.backends import array_namespace, is_scalar

def _normalize_field(a):
    """Extract the underlying array from a field for numpy compatibility.

    OldStyleField subclasses np.ndarray and can be used directly.
    NewStyleField stores its data in ``.data`` and is not ndarray-compatible.
    """
    if isinstance(a, NewStyleField):
        return a.data
    else:
        return a

def field_einsum(subscripts, *operands, **kwargs):
    '''Evaluates the Einstein summation convention on the operand fields.

    This function uses the same conventions as numpy.einsum(). The input
    subscript is multiplexed over each position in the grid. The grids of each of
    the input field operands don't have to match, but must have the same lengths.

    The subscripts must be written as you would for a single position in the grid.
    The function alters these subscripts to multiplex over the entire grid.

    .. caution::
        Some subscripts may yield no exception, even though they would fail for
        a single point in the grid. The output in these cases can not be trusted.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    operands : list of array_like or `Field`
        These are the arrays or fields for the operation.
    out : {ndarray, None}, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Default is False.

    Returns
    -------
    Field
        The calculated Field based on the Einstein summation convention.

    Raises
    ------
    ValueError
        If all of the fields don't have    the same grid size. If the number of
        operands is not equal to the number of subscripts specified.
    '''
    operand_is_field = [is_field(o) for o in operands]
    if not any(operand_is_field):
        return einsum(subscripts, *operands, **kwargs)

    field_sizes = [o.grid.size for i, o in enumerate(operands) if operand_is_field[i]]
    element_sizes = [o.shape[-1] for i, o in enumerate(operands) if operand_is_field[i]]

    if not np.allclose(field_sizes, field_sizes[0]) or not np.allclose(element_sizes, element_sizes[0]):
        raise ValueError('All fields must be the same size for a field_einsum().')

    # Decompose the subscript into input and output
    splitted_string = subscripts.split('->')
    if len(splitted_string) == 2:
        ss_input, ss_output = splitted_string
    else:
        ss_input = splitted_string[0]
        ss_output = ''

    # split the input operands in separate strings
    ss = ss_input.split(',')
    if len(ss) != len(operands):
        raise ValueError('Number of operands is not equal to number of indexing operands.')

    # Find an indexing letter that can be used for field dimension.
    unused_index = [a for a in string.ascii_lowercase if a not in subscripts][0]

    # Add the field dimension to the input field operands.
    ss = [s + unused_index if operand_is_field[i] else s for i, s in enumerate(ss)]

    # Recombine all operands into the final subscripts
    if len(splitted_string) == 2:
        subscripts_new = ','.join(ss) + '->' + ss_output + unused_index
    else:
        subscripts_new = ','.join(ss)

    unwrapped = [_normalize_field(o) if operand_is_field[i] else o for i, o in enumerate(operands)]

    if 'out' in kwargs:
        kwargs['out'] = _normalize_field(kwargs['out'])

    res = einsum(subscripts_new, *unwrapped, **kwargs)

    grid = None
    for o in operands:
        if hasattr(o, 'grid'):
            grid = o.grid
            break

    if 'out' in kwargs:
        kwargs['out'] = Field(res, grid)
    return Field(res, grid)

def field_dot(a, b, out=None):
    '''Perform a dot product of `a` and `b` multiplexed over the field dimension.

    Parameters
    ----------
    a : Field or array_like
        Left argument of the dot product.
    b : Field or array_like
        Right argument of the dot product.
    out : Field or array_like
        If provided, the calculation is done into this array.

    Returns
    -------
    Field
        The result of the dot product.
    '''
    # Find out if a or b are vectors or higher dimensional tensors
    if hasattr(a, 'tensor_order'):
        amat = a.tensor_order > 1
    elif is_scalar(a):
        if out is None:
            return a * b
        else:
            xp = array_namespace(a, b)
            return xp.multiply(a, b, out=out)
    else:
        amat = a.ndim > 1

    if hasattr(b, 'tensor_order'):
        bmat = b.tensor_order > 1
    elif is_scalar(b):
        if out is None:
            return a * b
        else:
            xp = array_namespace(a, b)
            return xp.multiply(a, b, out=out)
    else:
        bmat = b.ndim > 1

    # Select correct multiplication behaviour.
    if amat and bmat:
        subscripts = '...ij,...jk->...ik'
    elif amat and not bmat:
        subscripts = '...i,...i->...'
    elif not amat and bmat:
        subscripts = '...i,...ij->...j'
    elif not amat and not bmat:
        subscripts = '...i,...i->...'

    # Perform calculation and return.
    if out is None:
        return field_einsum(subscripts, a, b)
    else:
        return field_einsum(subscripts, a, b, out=out)

def field_trace(a, out=None):
    '''Take the trace of a tensor field.

    Parameters
    ----------
    a : Field
        The field of which to take the trace.
    out : Field or None
        If given, a location where the output is stored. This must have the correct shape.

    Returns
    -------
    Field
        The trace of the field `a`.
    '''
    if out is None:
        return field_einsum('ii', a)
    else:
        return field_einsum('ii', a, out=out)

def field_inverse_tikhonov(f, rcond=1e-15):
    '''Invert a tensor field of order 2 using Tikhonov regularization.

    Parameters
    ----------
    f : `Field` or ndarray
        The tensor field for which to calculate the inverses. The tensor order
        of this Field has to be 2. If it is not a field, a normal inverse with
        Tikhonov regularization will be returned.
    rcond : scalar
        The relative regularization parameter to use for the inversions.

    Returns
    -------
    Field or ndarray
        The resulting Field with tensor order 2.

    Raises
    ------
    ValueError
        If the tensor order of field `f` is not 2.
    '''
    from ..util import inverse_tikhonov

    if hasattr(f, 'grid'):
        if f.tensor_order != 2:
            raise ValueError("Field must be a tensor field of order 2 to be able to calculate inverses.")

        xp = array_namespace(_normalize_field(f))
        res = xp.empty((f.tensor_shape[1], f.tensor_shape[0], f.grid.size))
        for i in range(f.grid.size):
            res[..., i] = xp.asarray(inverse_tikhonov(_normalize_field(f[..., i]), rcond))
        return Field(res, f.grid)
    else:
        return inverse_tikhonov(f, rcond)

def field_inverse_truncated(f, rcond=1e-15):
    '''Invert a tensor field of order 2 using mode truncation.

    Parameters
    ----------
    f : `Field` or ndarray
        The tensor field for which to calculate the inverses. The tensor order
        of this Field has to be 2. If it is not a field, a normal truncated
        inverse will be returned.
    rcond : scalar
        The relative condition number of the highest-order mode that must
        be used for inversion.

    Returns
    -------
    Field or ndarray
        The resulting Field with tensor order 2.

    Raises
    ------
    ValueError
        If the tensor order of field `f` is not 2.
    '''
    from ..util import inverse_truncated

    if hasattr(f, 'grid'):
        if f.tensor_order != 2:
            raise ValueError("Field must be a tensor field of order 2 to be able to calculate inverses.")

        xp = array_namespace(_normalize_field(f))
        res = xp.empty((f.tensor_shape[1], f.tensor_shape[0], f.grid.size))
        for i in range(f.grid.size):
            res[..., i] = xp.asarray(inverse_truncated(_normalize_field(f[..., i]), rcond))
        return Field(res, f.grid)
    else:
        return inverse_truncated(f, rcond)

def field_inverse_truncated_modal(f, num_modes):
    '''Invert a tensor field of order 2 using mode truncation.

    Parameters
    ----------
    f : `Field` or ndarray
        The tensor field for which to calculate the inverses. The tensor order
        of this Field has to be 2. If it is not a field, a normal truncated
        inverse will be returned
    rcond : scalar
        The relative condition number of the highest-order mode that must
        be used for inversion.

    Returns
    -------
    Field or ndarray
        The resulting Field with tensor order 2.

    Raises
    ------
    ValueError
        If the tensor order of field `f` is not 2.
    '''
    from ..util import inverse_truncated_modal

    if hasattr(f, 'grid'):
        if f.tensor_order != 2:
            raise ValueError("Field must be a tensor field of order 2 to be able to calculate inverses.")

        xp = array_namespace(_normalize_field(f))
        res = xp.empty((f.tensor_shape[1], f.tensor_shape[0], f.grid.size))
        for i in range(f.grid.size):
            res[..., i] = xp.asarray(inverse_truncated_modal(_normalize_field(f[..., i]), num_modes))
        return Field(res, f.grid)
    else:
        return inverse_truncated_modal(f, num_modes)

def field_inv(f):
    '''Compute the (multiplicative) inverse of a tensor field of order 2.

    Parameters
    ----------
    f : `Field`
        The tensor field for which to calculate the inverses. The tensor order
        of this field has to be 2.

    Returns
    -------
    Field
        The resulting Field with tensor order 2.

    Raises
    ------
    ValueError
        If the tensor order of field `f` is not 2.
    '''
    if hasattr(f, 'grid'):
        if f.tensor_order != 2:
            raise ValueError('Field must be a tensor field of order 2 to be able to compute inverses.')

        data = _normalize_field(f)
        xp = array_namespace(data)
        res = xp.linalg.inv(xp.moveaxis(data, -1, 0))
        return Field(xp.moveaxis(res, 0, -1), f.grid)
    else:
        xp = array_namespace(f)
        return xp.linalg.inv(f)

def field_svd(f, full_matrices=True, compute_uv=True):
    '''Calculate the singular value decomposition for a tensor field of order 2.

    Parameters
    ----------
    f : `Field`
        The tensor field for which to calculate the singular value decompositions.
    full_matrices : boolean
        If True, matrices in U and Vh will have shapes (M,M) and (N,N) respectively.
        Otherwise their shapes are (M,K), (K,N) respectively, where K=min(M,N).
    compute_uv : boolean
        Whether to compute matrices U and Vh in addition to the singular values.

    Returns
    -------
    U : `Field`
        The unitary matrices. Only returned if `compute_uv` is True.
    S : `Field`
        The singular values, sorted in descending order.
    Vh : `Field`
        The unitary matrices. Only returned if `compute_uv` is True.
    '''

    data = _normalize_field(f)
    xp = array_namespace(data)

    if compute_uv:
        U, S, Vh = xp.linalg.svd(xp.moveaxis(data, -1, 0), full_matrices=full_matrices)
        U = Field(xp.moveaxis(U, 0, -1), f.grid)
        Vh = Field(xp.moveaxis(Vh, 0, -1), f.grid)
        S = Field(xp.moveaxis(S, 0, -1), f.grid)

        return U, S, Vh
    else:
        S = xp.linalg.svdvals(xp.moveaxis(data, -1, 0))
        S = Field(xp.moveaxis(S, 0, -1), f.grid)

        return S

def field_conjugate_transpose(a):
    '''Performs the conjugate transpose of a rank 2 tensor field or two dimensional array.

    Parameters
    ----------
    a : Field or array
        The element to conjugate transpose

    Returns
    -------
    Field or array
        The conjugate transposed element
    '''

    # first we test if it's a field
    if hasattr(a, 'tensor_order'):
        # if its a field, it must have a rank of 2
        if a.tensor_order != 2:
            raise ValueError('Need a tensor field of rank 2.')

        data = _normalize_field(a)
        xp = array_namespace(data)
        res = xp.permute_dims(xp.conj(data), (1, 0, 2))
        return Field(res, a.grid)
    else:
        # if its an array, it must be two dimensional
        if len(a.shape) != 2:
            raise ValueError('Need a two dimensional array.')

        xp = array_namespace(a)
        return xp.permute_dims(xp.conj(a), (1, 0))

def field_transpose(a):
    '''Performs the transpose of a rank 2 tensor field or two dimensional array.

    Parameters
    ----------
    a : Field or array
        The element to transpose

    Returns
    -------
    Field or array
        The transposed field or array
    '''
    # first we test if it's a field
    if hasattr(a, 'tensor_order'):
        # if its a field, it must have a rank of 2
        if a.tensor_order != 2:
            raise ValueError('Need a tensor field of rank 2.')

        data = _normalize_field(a)
        xp = array_namespace(data)
        res = xp.permute_dims(data, (1, 0, 2))
        return Field(res, a.grid)
    else:
        # if its an array, it must be two dimensional
        if len(a.shape) != 2:
            raise ValueError('Need a two dimensional array.')

        xp = array_namespace(a)
        return xp.permute_dims(a, (1, 0))

def field_determinant(a):
    '''Calculates the determinant of a tensor field.

    Parameters
    ----------
    a : Field
        The field for which the determinant needs to be calculated

    Returns
    -------
    Field
        The field that contains the determinant on every spatial position
    '''
    if a.tensor_order == 1:
        raise ValueError('Only tensor fields of order 2 or higher have a determinant.')

    if a.tensor_order > 2:
        raise NotImplementedError()

    if not all(n == a.tensor_shape[0] for n in a.tensor_shape):
        raise ValueError('Need square matrix for determinant.')

    # First we need to swap the axes in order to use xp.linalg.det
    data = _normalize_field(a)
    xp = array_namespace(data)
    Temp = xp.permute_dims(data, (2, 0, 1))

    return Field(xp.linalg.det(Temp), a.grid)

def field_adjoint(a):
    '''Calculates the adjoint of a tensor field.

    Parameters
    ----------
    a : Field
        The field for which the adjoint needs to be calculated

    Returns
    -------
    Field
        The adjointed field
    '''
    if a.tensor_order != 2:
        raise ValueError('Only tensor fields of order 2 can be inverted.')

    # Calculating the determinant.
    determinant = field_determinant(a)

    det_data = _normalize_field(determinant)
    xp = array_namespace(det_data)

    if xp.any(xp.abs(det_data) <= 1e-8):
        raise ValueError('Matrix is non-invertible due to zero determinant.')

    res = xp.reshape(det_data, (1, 1, -1)) * _normalize_field(field_inv(a))
    return Field(res, a.grid)

def field_cross(a, b):
    '''Calculates the cross product of two vector fields.

    Parameters
    ----------
    a : Field
        The first field of the cross product
    b : Field
        The second field of the cross product

    Returns
    -------
    Field
        The cross product field
    '''
    if a.tensor_order != 1 or b.tensor_order != 1:
        raise ValueError('Only tensor fields of order 1 can have a cross product.')

    if a.shape[0] != 3 or b.shape[0] != 3:
        raise ValueError('Vector needs to be of length 3 for cross product.')

    a_data = _normalize_field(a)
    b_data = _normalize_field(b)
    xp = array_namespace(a_data, b_data)

    return Field(xp.linalg.cross(a_data, b_data, axis=-2), a.grid)

def field_kron(a, b):
    '''Calculate the Kronecker product of two fields.

    Parameters
    ----------
    a : tensor Field
        The first Field
    b : tensor Field
        The second Field

    Returns
    -------
    Field
        The resulting tensor field.
    '''
    is_a_field = hasattr(a, 'grid')
    is_b_field = hasattr(b, 'grid')

    is_output_field = is_a_field or is_b_field

    if not is_output_field:
        return np.kron(a, b)

    if is_a_field and is_b_field:
        if a.grid.size != b.grid.size:
            raise ValueError('Field sizes for a (%d) and b (%d) are not compatible.' % (a.grid.size, b.grid.size))
        grid = a.grid
    else:
        if is_a_field:
            grid = a.grid
        else:
            grid = b.grid

    if is_a_field:
        aa = a
    else:
        aa = a[..., np.newaxis]

    if is_b_field:
        bb = b
    else:
        bb = b[..., np.newaxis]

    output_tensor_shape = np.array(aa.shape[:-1]) * np.array(bb.shape[:-1])
    output_shape = np.concatenate((output_tensor_shape, [grid.size]))

    res = (aa[:, np.newaxis, :, np.newaxis, :] * bb[np.newaxis, :, np.newaxis, :, :]).reshape(output_shape)

    return Field(res, grid)
