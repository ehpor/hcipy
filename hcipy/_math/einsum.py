import math
from collections import Counter
import numpy as np


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


def _prep_operand(sub, tensor, keep):
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
        tensor = np.diagonal(tensor, axis1=ax0, axis2=ax1)
        sub = [s for k, s in enumerate(sub) if k not in (ax0, ax1)] + [c]
    sub = "".join(sub)

    drop_axes = tuple(i for i, c in enumerate(sub) if c not in keep)
    if drop_axes:
        tensor = tensor.sum(axis=drop_axes)
        sub = "".join(c for c in sub if c in keep)
    return sub, tensor


def pairwise_einsum(l_sub, left, r_sub, right, needed):
    l_sub, left = _prep_operand(l_sub, left, set(r_sub) | needed)
    r_sub, right = _prep_operand(r_sub, right, set(l_sub) | needed)

    dims = dict(zip(l_sub + r_sub, left.shape + right.shape))

    B = [c for c in l_sub if c in r_sub and c in needed]      # batch
    K = [c for c in l_sub if c in r_sub and c not in needed]  # contracted
    M = [c for c in l_sub if c not in r_sub]                  # free, left-only
    N = [c for c in r_sub if c not in l_sub]                  # free, right-only

    Lp = np.transpose(left,  [l_sub.index(c) for c in B + M + K])
    Rp = np.transpose(right, [r_sub.index(c) for c in B + K + N])

    Lp = np.reshape(Lp, (math.prod(dims[c] for c in B), math.prod(dims[c] for c in M),
                         math.prod(dims[c] for c in K)))
    Rp = np.reshape(Rp, (math.prod(dims[c] for c in B), math.prod(dims[c] for c in K),
                         math.prod(dims[c] for c in N)))

    if not K:
        result = Lp * Rp
    else:
        result = np.matmul(Lp, Rp)

    result_sub = B + M + N
    result = np.reshape(result, tuple(dims[c] for c in result_sub))
    return "".join(result_sub), result


def einsum(subscripts, *operands, path=None):
    in_subs, out_sub = _parse_subscripts(subscripts, len(operands))
    subs = list(in_subs)
    tensors = [np.asarray(t) for t in operands]

    if path is None:
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
            new_sub, new_tensor = _prep_operand(
                popped_subs[0], popped_tensors[0], needed
            )
        else:
            new_sub, new_tensor = pairwise_einsum(
                popped_subs[0], popped_tensors[0],
                popped_subs[1], popped_tensors[1], needed
            )

        subs.append(new_sub)
        tensors.append(new_tensor)

    final_sub, final_tensor = subs[0], tensors[0]
    final_sub, final_tensor = _prep_operand(final_sub, final_tensor, set(out_sub))

    perm = [final_sub.index(c) for c in out_sub]
    return np.transpose(final_tensor, perm)
