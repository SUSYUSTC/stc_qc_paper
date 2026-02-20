import functools
import numba as nb
import numpy as np
import torch
from . import utils


def shape2stride(shape):
    stride = [1]
    for dim in shape[1:][::-1]:
        stride.append(stride[-1] * dim)
    return tuple(stride[::-1])


def unravel_index(indices, shape, return_tensor=False):
    strides = shape2stride(shape)
    # the last stride is always 1
    N = len(indices)
    n = len(shape)
    result = torch.empty((N, n), dtype=indices.dtype)
    for i, s in enumerate(strides):
        result[:, i] = indices // s
        indices = indices - result[:, i] * s
    if return_tensor:
        return result
    else:
        return tuple(result.T)


def ravel_multi_index(indices, shape):
    strides = shape2stride(shape)
    return sum(i * s for i, s in tuple(zip(indices, strides))[:-1]) + indices[-1]


@functools.partial(torch.compile, dynamic=True)
def get_below_thres(A, thres):
    return (torch.abs(A) < thres)


@functools.partial(torch.compile, dynamic=True)
def get_over_thres(A, thres):
    return (torch.abs(A) > thres)


@functools.partial(torch.compile, dynamic=True)
def sum_below_thres(A, thres):
    return torch.sum(A * (torch.abs(A) < thres))


@functools.partial(torch.compile, dynamic=True)
def sum_over_thres(A, thres):
    return torch.sum(A * (torch.abs(A) > thres))


@functools.partial(torch.compile, dynamic=True)
def count_below_thres(A, thres):
    return torch.sum(torch.abs(A) < thres)


@functools.partial(torch.compile, dynamic=True)
def count_over_thres(A, thres):
    return torch.sum(torch.abs(A) > thres)


@functools.partial(torch.compile, dynamic=True)
def linear_combination(Ts, Cs):
    assert isinstance(Ts, tuple)
    assert isinstance(Cs, tuple)
    assert len(Ts) == len(Cs)
    return sum(T * c for T, c in zip(Ts, Cs))


@functools.partial(torch.compile, dynamic=True)
def add_tensor_inplace(A, B, coeff):
    A += B * coeff


@functools.partial(torch.compile, dynamic=True)
def nd_dot(A, B, abs=False):
    ndim1 = A.ndim
    ndim2 = B.ndim
    sum_axes = tuple(range(ndim1 - ndim2, ndim1))
    if abs:
        return torch.abs(A * B).sum(dim=sum_axes)
    else:
        return (A * B).sum(dim=sum_axes)


def normalize_einsum(expr: str) -> str:
    # extract all letters in order of appearance
    letters = []
    for c in expr:
        if c.isalpha() and c not in letters:
            letters.append(c)
    # map them to a..z
    mapping = {c: chr(ord('a') + i) for i, c in enumerate(letters)}
    # replace using regex to avoid partial replacements
    return ''.join(mapping.get(c, c) for c in expr)


@functools.lru_cache(maxsize=None)
def get_compiled_einsum_func(expr, *, abs):
    def func(*args):
        if abs:
            args = [torch.abs(arg) for arg in args]
        return torch.einsum(expr, *args)
    return torch.compile(func, dynamic=True)


def fast_einsum(expr, *operands, abs=False):
    expr_in, expr_out = expr.split('->')
    exprs_in = expr_in.split(',')
    letters_all = set(''.join(exprs_in))
    free = set(expr_out)
    assert free.issubset(letters_all)
    dummy = letters_all - free
    assert len(free) > 0
    for letter in free:
        assert expr_in.count(letter) == 1
    for letter in dummy:
        assert expr_in.count(letter) == 2
    idx_full = None
    for i, e in enumerate(exprs_in):
        if set(e) == letters_all:
            idx_full = i
            break
    if (idx_full is not None) and (len(dummy) > 0):
        # matvec
        exprs_in_others = [e for i, e in enumerate(exprs_in) if i != idx_full]
        expr_others = ','.join(exprs_in_others) + '->' + ''.join(exprs_in_others)
        operands_others = [operands[i] for i in range(len(operands)) if i != idx_full]
        A = operands[idx_full]
        letters_A = exprs_in[idx_full]
        B = torch.einsum(expr_others, *operands_others)
        letters_B = ''.join(exprs_in_others)
        A = torch.einsum(f'{letters_A}->{expr_out}{letters_B}', A)
        #strides = [A.strides[letters_A.index(l)] for l in letters_B]
        #order = np.argsorted(strides)[::-1].tolist()
        #letters_B_ordered = ''.join(letters_B[i] for i in order)
        #A = torch.einsum(f'{letters_A}->{expr_out}{letters_B_ordered}', A)
        #B = torch.einsum(f'{letters_B}->{letters_B_ordered}', B).contiguous()
        return nd_dot(A, B, abs=abs)
    else:
        # matmat
        norm_expr = normalize_einsum(expr)
        if abs:
            einsum_func = get_compiled_einsum_func(norm_expr, abs=True)
        else:
            einsum_func = lambda *args: torch.einsum(norm_expr, *args)
        return einsum_func(*operands)


@utils.reference_decorate
def transform_from_right(T, Us):
    T = utils.pop_reference_data(T)
    assert T.is_contiguous()
    assert T.ndim == len(Us)
    for U in Us[::-1]:
        if U is None:
            T = torch.moveaxis(T, -1, 0).contiguous()
        else:
            # Uij TKj -> iK
            ni, nj = U.shape
            assert T.shape[-1] == nj
            K = T.shape[:-1]
            nk = K.numel()
            newshape = (ni, ) + K
            out = torch.zeros(newshape)
            torch.matmul(U, T.view((nk, nj)).T, out=out.view((ni, nk)))
            T = out
    return T


@utils.reference_decorate
def transform_from_left(T, Us):
    T = utils.pop_reference_data(T)
    assert T.is_contiguous()
    assert T.ndim == len(Us)
    for U in Us:
        if U is None:
            T = torch.moveaxis(T, 0, -1).contiguous()
        else:
            # Uij TjK -> Ki
            ni, nj = U.shape
            assert T.shape[0] == nj
            K = T.shape[1:]
            nk = K.numel()
            newshape = K + (ni, )
            out = torch.zeros(newshape)
            torch.matmul(T.view((nj, nk)).T, U.T, out=out.view((nk, ni)))
            T = out
    return T


def get_Linv(S):
    return torch.linalg.solve_triangular(torch.linalg.cholesky(S), torch.eye(S.shape[0]), upper=False)


def matrix_operation(A, func):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = func(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T
