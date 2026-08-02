import math
from collections import Counter

from .backends import array_namespace
from array_api_compat import is_numpy_namespace, is_cupy_namespace, is_jax_namespace, is_torch_namespace


def _parse_subscripts(subscripts, noperands):
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        in_part, out_part = subscripts.split("->")
    else:
        in_part, out_part = subscripts, None

    in_subs = in_part.split(",")
    if len(in_subs) != noperands:
        raise ValueError(f"got {noperands} operands but {len(in_subs)} subscripts")

    if out_part is None:
        # implicit output: every letter that appears exactly once, alphabetical
        counts = Counter(in_part.replace(",", ""))
        out_part = "".join(sorted(c for c in counts if counts[c] == 1))

    return in_subs, out_part


def _diagonal(tensor, ax0, ax1, xp):
    diag = getattr(xp, "diagonal", None)
    if diag is not None:
        return diag(tensor, 0, ax0, ax1)
    # array API fallback: move the two axes to the end, then mask with eye
    axes = [i for i in range(tensor.ndim) if i not in (ax0, ax1)] + [ax0, ax1]
    tensor = xp.permute_dims(tensor, axes)
    return xp.sum(tensor * xp.eye(tensor.shape[-1], dtype=tensor.dtype), axis=-1)


def _prep_operand(sub, tensor, keep, xp):
    sub = list(sub)
    while True:
        seen = {}
        pair = None
        for axis, c in enumerate(sub):
            if c in seen:
                pair = (seen[c], axis)
                break
            seen[c] = axis
        if pair is None:
            break
        ax0, ax1 = pair
        c = sub[ax0]
        tensor = _diagonal(tensor, ax0, ax1, xp)
        sub = [s for k, s in enumerate(sub) if k not in (ax0, ax1)] + [c]
    sub = "".join(sub)

    drop_axes = tuple(i for i, c in enumerate(sub) if c not in keep)
    if drop_axes:
        tensor = xp.sum(tensor, axis=drop_axes)
        sub = "".join(c for c in sub if c in keep)
    return sub, tensor


def pairwise_einsum(l_sub, left, r_sub, right, needed, xp):
    l_sub, left = _prep_operand(l_sub, left, set(r_sub) | needed, xp)
    r_sub, right = _prep_operand(r_sub, right, set(l_sub) | needed, xp)

    dims = dict(zip(l_sub + r_sub, left.shape + right.shape))

    B = [c for c in l_sub if c in r_sub and c in needed]      # batch
    K = [c for c in l_sub if c in r_sub and c not in needed]  # contracted
    M = [c for c in l_sub if c not in r_sub]                  # free, left-only
    N = [c for c in r_sub if c not in l_sub]                  # free, right-only

    Lp = xp.permute_dims(left, tuple(l_sub.index(c) for c in B + M + K))
    Rp = xp.permute_dims(right, tuple(r_sub.index(c) for c in B + K + N))

    Lp = xp.reshape(Lp, (math.prod(dims[c] for c in B), math.prod(dims[c] for c in M),
                         math.prod(dims[c] for c in K)))
    Rp = xp.reshape(Rp, (math.prod(dims[c] for c in B), math.prod(dims[c] for c in K),
                         math.prod(dims[c] for c in N)))

    if not K:
        result = Lp * Rp
    else:
        result = xp.matmul(Lp, Rp)

    result_sub = B + M + N
    result = xp.reshape(result, tuple(dims[c] for c in result_sub))
    return "".join(result_sub), result


def einsum(subscripts, *operands, optimize=True):
    xp = array_namespace(*operands)

    # Dispatch to the einsum of the backend, if it has one.
    if is_numpy_namespace(xp) or is_cupy_namespace(xp) or is_jax_namespace(xp):
        return xp.einsum(subscripts, *operands, optimize=optimize)

    if is_torch_namespace(xp):
        # Pytorch doesn't support the optimize keyword.
        return xp.einsum(subscripts, *operands)

    in_subs, out_sub = _parse_subscripts(subscripts, len(operands))
    subs = list(in_subs)
    tensors = [xp.asarray(t) for t in operands]

    if optimize:
        import opt_einsum as oe
        path, _ = oe.contract_path(subscripts, *tensors)
    else:
        path = [(0, 1)] * (len(tensors) - 1)

    for step in path:
        # pop high-to-low so positions don't shift; pairwise contraction is
        # commutative, so the pop order within a step doesn't matter
        step = sorted(step, reverse=True)
        if len(step) > 2:
            raise ValueError(f"path step {tuple(step)} pops more than 2 "
                             "operands; only 1- and 2-operand steps are "
                             "supported")
        popped_subs = [subs.pop(i) for i in step]
        popped_tensors = [tensors.pop(i) for i in step]

        # "needed" = final output letters + every letter still alive in any
        # tensor we haven't reached yet in the path
        needed = set(out_sub) | set("".join(subs))

        if len(popped_tensors) == 1:
            new_sub, new_tensor = _prep_operand(popped_subs[0], popped_tensors[0], needed, xp)
        else:
            new_sub, new_tensor = pairwise_einsum(
                popped_subs[0], popped_tensors[0],
                popped_subs[1], popped_tensors[1], needed, xp
            )

        subs.append(new_sub)
        tensors.append(new_tensor)

    final_sub, final_tensor = subs[0], tensors[0]
    final_sub, final_tensor = _prep_operand(final_sub, final_tensor, set(out_sub), xp)

    perm = tuple(final_sub.index(c) for c in out_sub)
    return xp.permute_dims(final_tensor, perm)
