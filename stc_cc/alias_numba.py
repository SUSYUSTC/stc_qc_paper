import functools
import numpy as np
import numba as nb

boundscheck = False
fastmath = False
njit_kwargs = dict(boundscheck=boundscheck, fastmath=fastmath, cache=True)
INT_2_31 = np.array(-2**31, dtype=np.int32)


@nb.njit(inline='never', **njit_kwargs)
def get_alias_numba(prob_in, prob, alias, this_sign, alias_sign):
    N = len(prob_in)
    #this_sign[:] = prob_in < 0
    #alias[:] = np.arange(N, dtype=np.int32)
    #updated = np.zeros((N, ), dtype=np.bool_)
    #total = np.abs(prob_in).sum()
    updated = np.zeros((N, ), dtype=np.bool_)
    total = 0.0
    for i in range(N):
        this_sign[i] = prob_in[i] < 0
        alias[i] = i
        total += abs(prob_in[i])
    if total == 0.0:
        prob[:] = 1.0
        alias_sign[:] = 0
        return
    scale = N / total
    for i in range(N):
        prob[i] = abs(prob_in[i]) * scale
    idx_small = -1
    idx_large = -1
    search_small = True
    search_large = True
    for _ in range(N - 1):
        if search_small:
            while True:
                idx_small += 1
                if (not updated[idx_small]) and prob[idx_small] <= 1.0:
                    break
            current_idx = idx_small
        else:
            current_idx = idx_large
        if search_large:
            while True:
                idx_large += 1
                if prob[idx_large] > 1.0:
                    break
        alias[current_idx] = idx_large
        new_prob_large = prob[idx_large] - (1.0 - prob[current_idx])
        prob[idx_large] = new_prob_large
        updated[current_idx] = True
        if (new_prob_large < 1.0):
            search_large = True
            search_small = False
        else:
            search_large = False
            search_small = True
    alias_sign[:] = this_sign[alias]


@functools.lru_cache(maxsize=None)
def _get_batched_alias_numba_1_unpacked(ndim_free):
    @nb.njit(parallel=True, **njit_kwargs)
    def func(data_out, M):
        prob_all, alias_all, this_sign_all, alias_sign_all = data_out
        row_shape = M.shape[0:ndim_free]
        for _idx in nb.pndindex(row_shape):
            idx = tuple(_idx)
            v = M[idx].flatten()
            get_alias_numba(v, prob_all[idx], alias_all[idx], this_sign_all[idx], alias_sign_all[idx])
    return func


def get_batched_alias_numba_1_unpacked(out_dim, data_out, M):
    ndim_free = M.ndim - out_dim
    func = _get_batched_alias_numba_1_unpacked(ndim_free)
    func(data_out, M)


@functools.lru_cache(maxsize=None)
def _get_batched_alias_numba_1_packed(ndim_free):
    @nb.njit(parallel=True, **njit_kwargs)
    def func(data_out, M):
        prob_all, alias_all = data_out
        row_shape = M.shape[0:ndim_free]
        for _idx in nb.pndindex(row_shape):
            idx = tuple(_idx)
            v = M[idx].flatten()
            ncol = len(v)
            this_sign = np.zeros((ncol,), dtype=np.bool_)
            alias_sign = np.zeros((ncol,), dtype=np.bool_)
            get_alias_numba(v, prob_all[idx], alias_all[idx], this_sign, alias_sign)
            prob_all[idx] = np.where(this_sign, -prob_all[idx], prob_all[idx])
            alias_all[idx] = np.where(alias_sign, alias_all[idx] + INT_2_31, alias_all[idx])
    return func


def get_batched_alias_numba_1_packed(out_dim, data_out, M):
    ndim_free = M.ndim - out_dim
    func = _get_batched_alias_numba_1_packed(ndim_free)
    func(data_out, M)


@functools.lru_cache(maxsize=None)
def _get_batched_alias_numba_2_unpacked(ndim1_free, ndim2_free):
    @nb.njit(parallel=True, **njit_kwargs)
    def func(data_out, M1, M2):
        prob_all, alias_all, this_sign_all, alias_sign_all = data_out
        row1_shape = M1.shape[0:ndim1_free]
        row2_shape = M2.shape[0:ndim2_free]
        for _idx1 in nb.pndindex(row1_shape):
            for _idx2 in np.ndindex(row2_shape):
                idx1 = tuple(_idx1)
                idx2 = tuple(_idx2)
                idx = idx1 + idx2
                v = (M1[idx1] * M2[idx2]).flatten()
                get_alias_numba(v, prob_all[idx], alias_all[idx], this_sign_all[idx], alias_sign_all[idx])
    return func


def get_batched_alias_numba_2_unpacked(out_dim, data_out, M1, M2):
    ndim1_free = M1.ndim - out_dim
    ndim2_free = M2.ndim - out_dim
    func = _get_batched_alias_numba_2_unpacked(ndim1_free, ndim2_free)
    func(data_out, M1, M2)


@functools.lru_cache(maxsize=None)
def _get_batched_alias_numba_2_packed(ndim1_free, ndim2_free):
    @nb.njit(parallel=True, **njit_kwargs)
    def func(data_out, M1, M2):
        prob_all, alias_all = data_out
        row1_shape = M1.shape[0:ndim1_free]
        row2_shape = M2.shape[0:ndim2_free]
        for _idx1 in nb.pndindex(row1_shape):
            for _idx2 in np.ndindex(row2_shape):
                idx1 = tuple(_idx1)
                idx2 = tuple(_idx2)
                idx = idx1 + idx2
                v = (M1[idx1] * M2[idx2]).flatten()
                ncol = len(v)
                this_sign = np.zeros((ncol,), dtype=np.bool_)
                alias_sign = np.zeros((ncol,), dtype=np.bool_)
                get_alias_numba(v, prob_all[idx], alias_all[idx], this_sign, alias_sign)
                prob_all[idx] = np.where(this_sign, -prob_all[idx], prob_all[idx])
                alias_all[idx] = np.where(alias_sign, alias_all[idx] + INT_2_31, alias_all[idx])
    return func


def get_batched_alias_numba_2_packed(out_dim, data_out, M1, M2):
    ndim1_free = M1.ndim - out_dim
    ndim2_free = M2.ndim - out_dim
    func = _get_batched_alias_numba_2_packed(ndim1_free, ndim2_free)
    func(data_out, M1, M2)


def get_batched_alias_numba(out_dim, data_out, M, M2=None, pack_sign=False):
    if M2 is None:
        func = get_batched_alias_numba_1_packed if pack_sign else get_batched_alias_numba_1_unpacked
        func(out_dim, data_out, M)
    else:
        assert M.ndim >= M2.ndim
        func = get_batched_alias_numba_2_packed if pack_sign else get_batched_alias_numba_2_unpacked
        func(out_dim, data_out, M, M2)


@nb.njit(**njit_kwargs)
def make_discrete_alias(nsamples_by_row):
    n = nsamples_by_row.shape[0]
    N = np.sum(nsamples_by_row)
    assert N % n == 0
    average = N // n

    bound = nsamples_by_row.copy()
    alias = np.arange(n, dtype=np.int32)
    small_idx = np.where(bound < average)[0].astype(np.int32)
    large_idx = np.where(bound > average)[0].astype(np.int32)
    nsmall = small_idx.shape[0]
    nlarge = large_idx.shape[0]
    idx_small = 0
    idx_large = 0
    search_small = True
    search_large = True
    while True:
        if search_small:
            if idx_small == nsmall:
                break
            else:
                i_small = small_idx[idx_small]
        if search_large:
            if idx_large == nlarge:
                break
            else:
                i_large = large_idx[idx_large]
        alias[i_small] = i_large
        bound[i_large] = bound[i_large] - (average - bound[i_small])
        if bound[i_large] < average:
            i_small = i_large
            idx_large += 1
            search_small = False
            search_large = True
        elif bound[i_large] > average:
            idx_small += 1
            search_small = True
            search_large = False
        else:
            idx_small += 1
            idx_large += 1
            search_small = True
            search_large = True
    return bound, alias, average
