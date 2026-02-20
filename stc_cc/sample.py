import functools
import typing as tp
from dataclasses import dataclass
import numpy as np
from . import utils, la
from . import alias_numba
import torch

align_alias = False
pack_sign = False
use_splitmix64 = True
torch_compile_dynamic = functools.partial(torch.compile, dynamic=True)

# quota_sampling = False if you want truely independent samples and unbiased variance estimation
# quota_sampling = True generates very small bias but typically much better runtime performance
quota_sampling = True

INT_2_31_torch = torch.from_numpy(alias_numba.INT_2_31)


# logical right shift for 64-bit values stored in int64 tensors
def urshift64(x: torch.Tensor, k: int) -> torch.Tensor:
    # arithmetic shift then clear the sign-extended 1s
    return (x >> k) & ((1 << (64 - k)) - 1)  # this mask is <= 2^63-1 for k>=1 → safe in int64


def get_random_state():
    return torch.randint(-2**63, 2**63 - 1, (), dtype=torch.int64)


def splitmix64(N, state: torch.Tensor):
    state = state + 0x9E3779B97F4A7C15
    z = state + torch.arange(N, dtype=torch.int64)
    z = z ^ urshift64(z, 30)
    z = z * 0xBF58476D1CE4E5B9
    z = z ^ urshift64(z, 27)
    z = z * 0x94D049BB133111EB
    z = z ^ urshift64(z, 31)
    return z, state


def fast_uniform(N, state: torch.Tensor):
    if not use_splitmix64:
        r = torch.rand((N, ), dtype=torch.float64)
        return r, state
    z, state = splitmix64(N, state)
    mant = urshift64(z, 11).to(torch.float64)
    u = mant * (1.0 / 9007199254740992.0)
    return u.to(torch.float64), state


def fast_randint(N, state: torch.Tensor, n: int):
    if not use_splitmix64:
        rand_int = torch.randint(0, n, (N, ), dtype=torch.int)
        return rand_int, state
    z, state = splitmix64(N, state)
    rand_int = z % n
    return rand_int.to(torch.int), state


def get_alias_dtype():
    if pack_sign:
        alias_dtype = np.dtype([
            ('prob',       np.float64),
            ('alias',      np.int32),
        ], align=True)
    else:
        alias_dtype = np.dtype([
            ('prob',       np.float64),
            ('alias',      np.int32),
            ('this_sign',  np.bool_),
            ('alias_sign', np.bool_),
        ], align=True)
    return alias_dtype


class Info:
    def __init__(self, out_dim, M1, M2=None):
        self.out_dim = out_dim
        self.row1_shape = M1.shape[0:-out_dim]
        self.nrow1 = self.row1_shape.numel()
        self.col_shape = M1.shape[-out_dim:]
        if M2 is not None:
            assert M1.shape[-out_dim:] == M2.shape[-out_dim:]
            self.row2_shape = M2.shape[0:-out_dim]
            self.row_shape = self.row1_shape + self.row2_shape
            self.nrow2 = self.row2_shape.numel()
        else:
            self.row_shape = self.row1_shape
        self.nrow = self.row_shape.numel()
        self.ncol = self.col_shape.numel()
        self.full_shape = self.row_shape + self.col_shape
        self.nfull = self.nrow * self.ncol

class BatchedSampler:
    def set_weights(self, weights):
        self.weights = weights

    def sample_indices(self, indices_row, random_state, return_sign=False):
        raise NotImplementedError()


@torch_compile_dynamic
def get_batched_cdf_torch(M, cdf):
    M = M.reshape(cdf.shape)
    torch.cumsum(torch.abs(M), dim=1, out=cdf)
    cdf /= cdf[:, -1][:, None]


def searchsorted_by_row_idx(A, row_idx, r):
    return torch.searchsorted(A.index_select(0, row_idx.unsqueeze(0)).squeeze(0), r)


class BatchedCDFSampler(BatchedSampler):
    def __init__(self, out_dim, M):
        info = Info(out_dim, M)
        self.out_dim = out_dim
        self.cdf = torch.zeros(info.full_shape)
        get_batched_cdf_torch(M, self.cdf.view((info.nrow, info.ncol)))
        self.sign = self.cdf < 0

    def flatten(self):
        data = (self.cdf, self.sign, self.weights)
        aux_data = (self.out_dim, )
        return data, aux_data
    
    @classmethod
    def unflatten(cls, data, aux_data):
        out_dim, = aux_data
        cdf, sign, weights = data
        obj = cls.__new__(cls)
        obj.cdf = cdf
        obj.sign = sign
        obj.weights = weights
        obj.out_dim = out_dim
        return obj

    def sample_indices(self, uindices_row, random_state, return_sign=False):
        info = Info(self.out_dim, self.cdf)
        nsamples = len(uindices_row[0])
        r, random_state = fast_uniform(nsamples, random_state)
        cdf_selected = self.cdf[uindices_row].view((nsamples, info.ncol))
        indices_col = torch.searchsorted(cdf_selected, r[:, None]).squeeze(1)
        uindices_col = la.unravel_index(indices_col, info.col_shape)
        if return_sign:
            sign = self.sign[uindices_row + uindices_col]
            return uindices_col, sign, random_state
        else:
            return uindices_col, random_state


class BatchedAliasSampler(BatchedSampler):
    @utils.maybe_profile
    def __init__(self, out_dim, M1, M2=None):
        info = Info(out_dim, M1, M2)
        self.out_dim = out_dim
        self.pack_sign = pack_sign
        self.alias_dtype = get_alias_dtype()
        self.alias_names = self.alias_dtype.names
        self.alias_types = tuple([self.alias_dtype.fields[name][0] for name in self.alias_names])
        if M2 is not None:
            Ms_numpy = (M1.numpy(), M2.numpy())
        else:
            Ms_numpy = (M1.numpy(), )
        shape_data = info.row_shape + (info.ncol, )
        if align_alias:
            self.data_numpy = np.zeros(shape_data, dtype=self.alias_dtype)
            self.data = tuple(self.data_numpy[name] for name in self.alias_names)
            data_out = tuple(self.data_numpy[name] for name in self.alias_names)
        else:
            self.data_numpy = self.data = tuple(np.zeros(shape_data, dtype=dtype) for dtype in self.alias_types)
            data_out = self.data_numpy
        self.data = tuple(torch.from_numpy(arr).view(info.full_shape) for arr in self.data)
        alias_numba.get_batched_alias_numba(self.out_dim, data_out, *Ms_numpy, pack_sign=self.pack_sign)
        self.weights = None
        self.updated = True

    def unpack(self, data):
        if self.pack_sign:
            packed_prob, packed_alias = data
            this_sign = packed_prob < 0
            alias_sign = packed_alias < 0
            prob = torch.abs(packed_prob)
            alias = torch.where(alias_sign, packed_alias + INT_2_31_torch, packed_alias)
            return prob, alias, this_sign, alias_sign
        else:
            return data

    def flatten(self):
        data = self.data + (self.weights, )
        aux_data = (self.pack_sign, self.out_dim, )
        return data, aux_data

    @classmethod
    def unflatten(cls, data, aux_data):
        weights = data[-1]
        data = data[0:-1]
        pack_sign, out_dim = aux_data
        obj = cls.__new__(cls)
        obj.data = data
        obj.weights = weights
        obj.pack_sign = pack_sign
        obj.out_dim = out_dim
        return obj

    def sample_indices(self, uindices_row, random_state: torch.Tensor, return_sign=False):
        info = Info(self.out_dim, self.data[0])
        nsamples = len(uindices_row[0])
        assert all(len(idx) == nsamples for idx in uindices_row)
        trial_col, random_state = fast_randint(nsamples, random_state, info.ncol)
        r, random_state = fast_uniform(nsamples, random_state)

        shape_pack = info.row_shape + (info.ncol, )
        uindices_pack = uindices_row + (trial_col, )

        packed_data = tuple([item.view(shape_pack)[uindices_pack] for item in self.data])
        prob, alias, this_sign, alias_sign = self.unpack(packed_data)

        use_alias = (r >= prob)
        indices_col = torch.where(use_alias, alias, trial_col)
        uindices_col = la.unravel_index(indices_col, info.col_shape)
        if return_sign:
            sign = torch.where(use_alias, alias_sign, this_sign)
            return uindices_col, sign, random_state
        else:
            return uindices_col, random_state


@torch_compile_dynamic
def get_block_probs(out_dim, M_by_block, M2_by_block=None, square=False):
    process_func = torch.square if square else torch.abs
    if M2_by_block is None:
        block_probs = process_func(M_by_block).sum(dim=-1)
        return block_probs.contiguous()
    else:
        ndim1 = M_by_block.ndim - out_dim - 1
        ndim2 = M2_by_block.ndim - out_dim - 1
        assert ndim1 >= ndim2
        if ndim2 == 0:
            return process_func(M_by_block * M2_by_block).sum(dim=-1).contiguous()
        else:
            expr1 = 'ijklmn'[:ndim1]
            expr2 = 'abcdef'[:ndim2]
            expr_shared = 'pqrstu'[:out_dim]
            einsum_str = f"{expr1}{expr_shared}x,{expr2}{expr_shared}x->{expr1}{expr2}{expr_shared}"
            block_probs = torch.einsum(einsum_str, process_func(M_by_block), process_func(M2_by_block))
            return block_probs.contiguous()


def get_block_probs_timing(out_dim, M_by_block, M2_by_block=None, square=False):
    import time
    t1 = time.time()
    block_probs = get_block_probs(out_dim, M_by_block, M2_by_block, square=square)
    t2 = time.time()

    def get_order(A):
        return tuple(np.argsort(A.stride())[::-1].tolist())

    if M2_by_block is not None:
        print(out_dim, M_by_block.shape, get_order(M_by_block), M2_by_block.shape, get_order(M2_by_block), t2 - t1)
    else:
        print(out_dim, M_by_block.shape, get_order(M_by_block), t2 - t1)
    return block_probs


class BatchedBlockSampler(BatchedSampler):
    @utils.maybe_profile
    def __init__(self, out_dim, block_size, alias, M1, M2=None, square=False):
        self.out_dim = out_dim
        self.block_size = block_size
        self.alias = alias
        self.square = square
        self.M1 = M1
        self.M2 = M2

        M1_by_block_front = M1[..., 0:self.nfront].reshape(M1.shape[0:-1] + (self.nblocks - 1, self.block_size))
        M1_by_block_end = M1[..., self.nfront:]
        if M2 is not None:
            assert M1.shape[-out_dim:] == M2.shape[-out_dim:]
            M2_by_block_front = M2[..., 0:self.nfront].reshape(M2.shape[0:-1] + (self.nblocks - 1, self.block_size))
            M2_by_block_end = M2[..., self.nfront:]
        else:
            M2_by_block_front = None
            M2_by_block_end = None
        block_probs_front = get_block_probs(self.out_dim, M1_by_block_front, M2_by_block_front, square=self.square)
        block_probs_end = get_block_probs(self.out_dim - 1, M1_by_block_end, M2_by_block_end, square=self.square)
        self.block_probs = torch.cat([block_probs_front, block_probs_end[..., None]], dim=-1)
        constructor = BatchedAliasSampler if alias else BatchedCDFSampler
        self.sub_sampler = constructor(self.out_dim, self.block_probs)
        self.weights = None

    @property
    def n(self):
        return self.M1.shape[-1]

    @property
    def nblocks(self):
        return utils.get_nbatches(self.n, self.block_size)

    @property
    def nfront(self):
        return (self.nblocks - 1) * self.block_size

    def flatten(self):
        sub_sampler_data, sub_sampler_aux = self.sub_sampler.flatten()
        data = (self.M1, self.M2, self.block_probs, sub_sampler_data, self.weights)
        aux_data = (self.sub_sampler.__class__, sub_sampler_aux, self.out_dim, self.square)
        return data, aux_data

    @classmethod
    def unflatten(cls, data, aux_data):
        M1, M2, block_probs, sub_sampler_data, weights = data
        sub_sampler_class, sub_sampler_aux, out_dim, square = aux_data
        sub_sampler = sub_sampler_class.unflatten(sub_sampler_data, sub_sampler_aux)
        obj = cls.__new__(cls)
        obj.out_dim = out_dim
        obj.square = square
        obj.M1 = M1
        obj.M2 = M2
        obj.block_probs = block_probs
        obj.sub_sampler = sub_sampler
        obj.weights = weights
        return obj

    def get_M_selected(self, uindices_M, indices_inner=None):
        if indices_inner is None:
            indices_inner = slice(None)
        if self.M2 is None:
            uindices_full = uindices_M + (indices_inner, )
            return self.M1[uindices_full]
        else:
            ndim1 = self.M1.ndim - self.out_dim
            ndim2 = self.M2.ndim - self.out_dim
            assert len(uindices_M) == ndim1 + ndim2 + self.out_dim - 1
            uindices_row1 = uindices_M[:ndim1]
            uindices_row2 = uindices_M[ndim1:ndim1+ndim2]
            uindices_block = uindices_M[ndim1+ndim2:]
            uindices_full1 = uindices_row1 + uindices_block + (indices_inner, )
            uindices_full2 = uindices_row2 + uindices_block + (indices_inner, )
            return self.M1[uindices_full1], self.M2[uindices_full2]

    def sample_indices(self, uindices_row, random_state, return_sign=False):
        nsamples = len(uindices_row[0])
        assert all(len(idx) == nsamples for idx in uindices_row)
        uindices_aliascol, random_state = self.sub_sampler.sample_indices(uindices_row, random_state)
        uindices_col = uindices_aliascol[0:-1]
        indices_block = uindices_aliascol[-1]
        uindices_M = uindices_row + uindices_col
        r, random_state = fast_uniform(nsamples, random_state)
        block_probs_selected = self.block_probs[uindices_M + (indices_block, )] * r

        indices_begin = indices_block * self.block_size - self.n
        cumsum = block_probs_selected
        indices_inner = torch.zeros_like(indices_begin)

        process_func = torch.square if self.square else torch.abs
        if self.M2 is None:
            for i in range(self.block_size - 1):
                values = self.get_M_selected(uindices_M, indices_begin + i)
                cumsum -= process_func(values)
                indices_inner += (cumsum > 0)
            indices_last = indices_begin + indices_inner + self.n
            if return_sign:
                values = self.get_M_selected(uindices_M, indices_last)
                sign = values < 0
            else:
                sign = None
        else:
            for i in range(self.block_size - 1):
                values1, values2 = self.get_M_selected(uindices_M, indices_begin + i)
                cumsum -= process_func(values1 * values2)
                indices_inner += (cumsum > 0)
            indices_last = indices_begin + indices_inner + self.n
            if return_sign:
                values1, values2 = self.get_M_selected(uindices_M, indices_last)
                sign = (values1 < 0) ^ (values2 < 0)
            else:
                sign = None
        uindices_fullcol = uindices_col + (indices_last, )
        if return_sign:
            return uindices_fullcol, sign, random_state
        else:
            return uindices_fullcol, random_state


class AlignedBatchedBlockSampler(BatchedSampler):
    def __init__(self, out_dim, block_size, alias, M1, M2=None, square=False):
        self.out_dim = out_dim
        block_size = block_size
        self.alias = alias
        self.square = square
        assert M1.shape[-1] % block_size == 0
        self.M1_by_block = M1.reshape(M1.shape[0:-1] + (-1, block_size))
        if M2 is not None:
            assert M1.shape[-out_dim:] == M2.shape[-out_dim:]
            self.M2_by_block = M2.reshape(M2.shape[0:-1] + (-1, block_size))
        else:
            self.M2_by_block = None
        self.block_probs = get_block_probs(self.out_dim, self.M1_by_block, self.M2_by_block, square=self.square)
        constructor = BatchedAliasSampler if alias else BatchedCDFSampler
        self.sub_sampler = constructor(self.out_dim, self.block_probs)
        self.weights = None

    def flatten(self):
        sub_sampler_data, sub_sampler_aux = self.sub_sampler.flatten()
        data = (self.M1_by_block, self.M2_by_block, self.block_probs, sub_sampler_data, self.weights)
        aux_data = (self.sub_sampler.__class__, sub_sampler_aux, self.out_dim, self.square)
        return data, aux_data

    @classmethod
    def unflatten(cls, data, aux_data):
        M_by_block, M2_by_block, block_probs, sub_sampler_data, weights = data
        sub_sampler_class, sub_sampler_aux, out_dim, square = aux_data
        sub_sampler = sub_sampler_class.unflatten(sub_sampler_data, sub_sampler_aux)
        obj = cls.__new__(cls)
        obj.out_dim = out_dim
        obj.square = square
        obj.M_by_block = M_by_block
        obj.M2_by_block = M2_by_block
        obj.block_probs = block_probs
        obj.sub_sampler = sub_sampler
        obj.weights = weights
        return obj

    def get_M_selected(self, uindices_M, indices_inner=None):
        if indices_inner is None:
            indices_inner = slice(None)
        if self.M2_by_block is None:
            uindices_full = uindices_M + (indices_inner, )
            return self.M1_by_block[uindices_full]
        else:
            ndim1 = self.M1_by_block.ndim - self.out_dim - 1
            #ndim2 = self.M2_by_block.ndim - self.out_dim - 1
            uindices_row = uindices_M[:-self.out_dim]
            uindices_block = uindices_M[-self.out_dim:]
            uindices_full1 = uindices_row[:ndim1] + uindices_block + (indices_inner, )
            uindices_full2 = uindices_row[ndim1:] + uindices_block + (indices_inner, )
            return self.M1_by_block[uindices_full1], self.M2_by_block[uindices_full2]

    def get_indices_inner_loop(self, block_probs_selected, M_selected, M2_selected=None):
        nsamples = M_selected.shape[0]
        block_size = self.M1_by_block.shape[-1]
        indices_inner = torch.zeros((nsamples, ), dtype=torch.int)
        process_func = torch.square if self.square else torch.abs
        for i in range(block_size - 1):
            values = M_selected[:, i] if M2_selected is None else M_selected[:, i] * M2_selected[:, i]
            block_probs_selected -= process_func(values)
            indices_inner += block_probs_selected > 0
        return indices_inner

    def sample_indices(self, uindices_row, random_state, return_sign=False):
        nsamples = len(uindices_row[0])
        assert all(len(idx) == nsamples for idx in uindices_row)
        uindices_block, random_state = self.sub_sampler.sample_indices(uindices_row, random_state)
        uindices_M = uindices_row + uindices_block
        r, random_state = fast_uniform(nsamples, random_state)
        block_probs_selected = self.block_probs[uindices_M] * r
        arange = torch.arange(nsamples, dtype=torch.int64)

        block_size = self.M1_by_block.shape[-1]
        if self.M2_by_block is None:
            M_selected = self.get_M_selected(uindices_M)
            indices_inner = self.get_indices_inner_loop(block_probs_selected, M_selected)
            if return_sign:
                sign = M_selected[arange, indices_inner] < 0
            else:
                sign = None
        else:
            M1_selected, M2_selected = self.get_M_selected(uindices_M)
            indices_inner = self.get_indices_inner_loop(block_probs_selected, M1_selected, M2_selected)
            if return_sign:
                sign = (M1_selected[arange, indices_inner] < 0) ^ (M2_selected[arange, indices_inner] < 0)
            else:
                sign = None
        indices_last = uindices_block[-1] * block_size + indices_inner
        uindices_col = uindices_block[:-1] + (indices_last, )
        if return_sign:
            return uindices_col, sign, random_state
        else:
            return uindices_col, random_state


torch.utils._pytree.register_pytree_node(BatchedCDFSampler, BatchedCDFSampler.flatten, BatchedCDFSampler.unflatten)
torch.utils._pytree.register_pytree_node(BatchedAliasSampler, BatchedAliasSampler.flatten, BatchedAliasSampler.unflatten)
torch.utils._pytree.register_pytree_node(BatchedBlockSampler, BatchedBlockSampler.flatten, BatchedBlockSampler.unflatten)
torch.utils._pytree.register_pytree_node(AlignedBatchedBlockSampler, AlignedBatchedBlockSampler.flatten, AlignedBatchedBlockSampler.unflatten)


@torch_compile_dynamic
def sample_indices(sampler: BatchedSampler, uindices_row, random_state: torch.Tensor, return_sign=False):
    return sampler.sample_indices(uindices_row, random_state, return_sign=return_sign)


@dataclass
class EinsumData:
    expr: str
    path: tp.List[str]
    effective_coeff: torch.Tensor


torch.utils._pytree.register_dataclass(EinsumData)


def general_sampling(data: EinsumData, samplers: tp.List[BatchedSampler], uindices_marginal, nsamples_tot, random_state: torch.Tensor):
    expr_out = data.expr.split('->')[-1]
    indices_dict = {}

    def update_indices_dict(letters, uindices):
        nonlocal indices_dict
        for letter, indices in zip(letters, uindices):
            indices_dict[letter] = indices

    nsamples = uindices_marginal[0].shape[0]
    marginal_letters = data.path[0].split('->')[0]
    update_indices_dict(marginal_letters, uindices_marginal)

    total_signs = torch.zeros(nsamples, dtype=bool)
    coeff = data.effective_coeff / nsamples_tot
    #values = torch.full((nsamples, ), coeff, dtype=torch.float64)
    values = torch.ones((nsamples, )) * coeff
    for path_item, sampler in zip(data.path, samplers):
        row_letters, col_letters = path_item.split('->')
        uindices_row = tuple(indices_dict[letter] for letter in row_letters)
        uindices_col, signs, random_state = sampler.sample_indices(uindices_row, random_state, return_sign=True)
        total_signs ^= signs
        if sampler.weights is not None:
            values *= sampler.weights[uindices_col]
        update_indices_dict(col_letters, uindices_col)
    values *= torch.where(total_signs, -1, 1)
    uindices_out = tuple(indices_dict[letter] for letter in expr_out)
    return uindices_out, values


def perform_sampling_add(data: EinsumData, samplers: tp.List[BatchedSampler], uindices_marginal, nsamples_tot, random_state, out):
    uindices_out, values = general_sampling(data, samplers, uindices_marginal, nsamples_tot, random_state)
    out.index_put_(uindices_out, values, accumulate=True)


def estimate_sampling_stats(data: EinsumData, samplers: tp.List[BatchedSampler], uindices_marginal, random_state, weights_out):
    uindices_out, values = general_sampling(data, samplers, uindices_marginal, torch.tensor(1.0), random_state)
    values_weights_out = weights_out[uindices_out]
    es = values * values_weights_out
    return es, values


perform_sampling_add_compile = utils.compile(perform_sampling_add)
estimate_sampling_stats_compile = utils.compile(estimate_sampling_stats)


@torch_compile_dynamic
def make_marginal_nsamples_quota(nsamples, p):
    shape = p.shape
    p = p.reshape(-1)
    N = len(p)
    norm = torch.sum(p)
    cumsum = torch.zeros(N + 1)
    torch.cumsum(p, 0, out=cumsum[1:])
    norm = cumsum[-1].clone()
    cumsum /= norm
    cumsum *= nsamples
    cumsum = cumsum.to(torch.int64)
    assert cumsum[-1] == nsamples
    counts = torch.diff(cumsum)
    return counts.reshape(shape)


def make_marginal_nsamples_exact(nsamples, p):
    counts = torch.distributions.Multinomial(probs=p.flatten(), total_count=nsamples).sample().reshape(p.shape).to(torch.int64)
    return counts


def make_marginal_nsamples(nsamples, p):
    if quota_sampling:
        return make_marginal_nsamples_quota(nsamples, p)
    else:
        return make_marginal_nsamples_exact(nsamples, p)


@torch_compile_dynamic
def make_marginal_indices(counts, nsamples, offset=0):
    shape = counts.shape
    N = shape.numel()

    indices_arange = torch.arange(offset, N + offset, dtype=torch.int)

    utensor_arange = la.unravel_index(indices_arange, shape, return_tensor=True)
    assert utensor_arange.is_contiguous()
    utensor = torch.repeat_interleave(utensor_arange, counts.reshape(-1), dim=0, output_size=nsamples)
    uindices = tuple(utensor.T)

    #indices = torch.repeat_interleave(indices_arange, counts.reshape(-1), output_size=nsamples)
    #uindices = la.unravel_index(indices, shape)
    return uindices


def make_batches(counts, batch_size):
    '''
    return a list of (begin, end) such that counts[begin:end].sum() >= batch_size for smallest possible end
    '''
    begin = 0
    batches = []
    N = len(counts)
    while begin < N:
        end = begin + 1
        cumsum = counts[begin].item()
        while end < N and cumsum < batch_size:
            cumsum += counts[end]
            end += 1
        batches.append((begin, end))
        begin = end
    return batches


def make_marginal_indices_by_batch(counts, batch_size):
    row_offset = counts.shape[1:].numel()
    if counts.ndim > 1:
        counts_grouped = counts.sum(dim=tuple(range(1, counts.ndim)))
    else:
        counts_grouped = counts
    idx_batches = make_batches(counts_grouped, batch_size)
    for begin, end in idx_batches:
        yield make_marginal_indices(counts[begin:end], counts_grouped[begin:end].sum().item(), row_offset * begin)
