import time
import functools
from . import la, sample, utils
import numpy as np
import torch
nbatch_vvov = 8
nsamples_test = 1000
nrun_estimation = 8
std_tol = 0.02


def r3(w):
    left = tuple(range(w.ndim))[3:]
    return (4 * w + w.permute((1, 2, 0) + left) + w.permute((2, 0, 1) + left)
            - 2 * w.permute((2, 1, 0) + left) - 2 * w.permute((0, 2, 1) + left)
            - 2 * w.permute((1, 0, 2) + left))


def get_index_batch(I, J, K, A, B, C, begin, end):
    mask = (I >= begin) & (I < end)
    return I[mask] - begin, J[mask], K[mask], A[mask], B[mask], C[mask], mask


def symm_indices(ijkabc):
    I, J, K, A, B, C = ijkabc
    return [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B), 
            (I, K, J, A, C, B), (J, I, K, B, A, C), (K, J, I, C, B, A)]


def r3_indices_weights(ijkabc):
    I, J, K, A, B, C = ijkabc
    return [((I, J, K, A, B, C), +4), ((J, K, I, A, B, C), +1), ((K, I, J, A, B, C), +1), 
            ((I, K, J, A, B, C), -2), ((J, I, K, A, B, C), -2), ((K, J, I, A, B, C), -2)]


def symmetrize(tensor, coeff1, coeff2, index1, index2):
    return la.linear_combination((tensor, tensor.swapaxes(index1, index2)), (coeff1, coeff2))


def sum_square(tensor, dim):
    return torch.sum(tensor**2, dim=dim)


@torch.compile(dynamic=True)
def SDDMM(A, B, I, J, square=False):
    prod = A[I] * B[J]
    if square:
        prod = prod ** 2
    return torch.sum(prod, dim=-1)


def divide_by_batch(array, batch, *, axis):
    if batch is None:
        yield array
    else:
        N = array.shape[axis]
        current = 0
        while current < N:
            slic = [slice(None) for _ in range(array.ndim)]
            slic[axis] = slice(current, current+batch)
            yield array[tuple(slic)]
            current += batch


def df_ccsd_t(t1, t2, R, e):
    import itertools
    orders_3 = np.array(list(itertools.permutations(range(3))))
    orders_6 = np.concatenate([orders_3, orders_3 + 3], axis=1)
    orders_6 = orders_6.tolist()

    nocc, _ = t1.shape
    o = slice(None, nocc)
    v = slice(nocc, None)
    e_occ = e[o]
    e_vir = e[v]
    e_occ, e_vir = e[o], e[v]
    Roo = R[o, o].clone(memory_format=torch.contiguous_format)
    Rvo = R[v, o].clone(memory_format=torch.contiguous_format)
    Rvv = R[v, v].clone(memory_format=torch.contiguous_format)
    eris_vvov = torch.einsum('aix,bcx->aibc', Rvo, Rvv)
    eris_vvoo = torch.einsum('aix,bjx->aibj', Rvo, Rvo)
    eris_vooo = torch.einsum('aix,jkx->aijk', Rvo, Roo)
    w1 = torch.einsum('aibf,kcjf->ijkabc', eris_vvov, t2)

    w2 = torch.einsum('aijm,mbkc->ijkabc', eris_vooo, t2)
    w = w1 - w2

    v = torch.einsum('aibj,kc->ijkabc', eris_vvoo, t1)

    eijk = e_occ[:, None, None] + e_occ[None, :, None] + e_occ[None, None, :]
    eabc = e_vir[:, None, None] + e_vir[None, :, None] + e_vir[None, None, :]
    d3 = eijk[:, :, :, None, None, None] - eabc[None, None, None, :, :, :]
    z = r3(w + v / 2)

    w_symm = sum([w.permute(tuple(order)) for order in orders_6])
    return torch.sum(w_symm * (z / d3)) * 2


@functools.partial(torch.compile, dynamic=True)
def fill_vvov(eris_vvov, Rvo, Rvv, x, y):
    data_triu = torch.einsum('aix,Bx->aiB', Rvo, Rvv[x, y])
    eris_vvov[:, :, x, y] = data_triu
    eris_vvov[:, :, y, x] = data_triu


def evaluate_vvov(Rvo, Rvv):
    nvir, nocc, _ = Rvo.shape
    x_upper, y_upper = torch.triu_indices(nvir, nvir, offset=1)
    x_batches = divide_by_batch(x_upper, nocc * nvir, axis=0)
    y_batches = divide_by_batch(y_upper, nocc * nvir, axis=0)
    eris_vvov = torch.zeros((nvir, nocc, nvir, nvir))
    for x, y in zip(x_batches, y_batches):
        fill_vvov(eris_vvov, Rvo, Rvv, x, y)
    diag = torch.arange(nvir)
    data_diag = torch.einsum('aix,Bx->aiB', Rvo, Rvv[diag, diag])
    eris_vvov[:, :, diag, diag] = data_diag
    return eris_vvov


class StochasticCCSDT(utils.Logging):
    def __init__(self, R, e, nocc, verbose=0, *, block_size, aligned):
        self.R = R
        self.e = e
        self.nao = len(e)
        self.nocc = nocc
        self.nvir = self.nao - nocc
        self.o = slice(None, self.nocc)
        self.v = slice(self.nocc, None)
        self.e_occ, self.e_vir = self.e[self.o], self.e[self.v]
        self.Roo = self.R[self.o, self.o].clone(memory_format=torch.contiguous_format)
        self.Rvo = self.R[self.v, self.o].clone(memory_format=torch.contiguous_format)
        self.Rvv = self.R[self.v, self.v].clone(memory_format=torch.contiguous_format)
        self.verbose = verbose
        self.occ_block_size, self.vir_block_size = block_size
        self.occ_aligned, self.vir_aligned = aligned
        if self.occ_aligned:
            assert self.nocc % self.occ_block_size == 0
            self.constructor_occ = sample.AlignedBatchedBlockSampler
        else:
            self.constructor_occ = sample.BatchedBlockSampler
        if self.vir_aligned:
            assert self.nvir % self.vir_block_size == 0
            self.constructor_vir = sample.AlignedBatchedBlockSampler
        else:
            self.constructor_vir = sample.BatchedBlockSampler

    @utils.maybe_profile
    def initialize(self):
        t1 = time.time()
        self.eris_vvov = evaluate_vvov(self.Rvo, self.Rvv)
        t2 = time.time()
        self.log(2, 'vvov time', t2 - t1)
        #self.eris_vvov = torch.einsum('aix,bcx->aibc', self.Rvo, self.Rvv)
        self.eris_oovv = torch.einsum('aix,bjx->iajb', self.Rvo, self.Rvo)
        t3 = time.time()
        self.log(2, 'oovv time', t3 - t2)
        self.eris_vooo = torch.einsum('aix,jkx->aijk', self.Rvo, self.Roo)
        t4 = time.time()
        self.log(2, 'vooo time', t4 - t3)
        assert self.eris_vvov.is_contiguous()
        assert self.eris_oovv.is_contiguous()
        assert self.eris_vooo.is_contiguous()
        Rvo_xx = torch.einsum('aix,aiy->xy', self.Rvo, self.Rvo)
        self.w1_fV = torch.einsum('bfx,bfy,xy->f', self.Rvv, self.Rvv, Rvo_xx)
        self.w2_mV = torch.einsum('jmx,jmy,xy->m', self.Roo, self.Roo, Rvo_xx)
        t5 = time.time()
        self.log(2, 'marginal time', t5 - t4)
        self.sampler_aib = self.constructor_vir(3, self.vir_block_size, True, self.eris_vvov.moveaxis(-2, 0), square=True)
        t6 = time.time()
        self.log(2, 'vvov sampler time', t6 - t5)
        self.sampler_aij = self.constructor_occ(3, self.occ_block_size, True, self.eris_vooo.moveaxis(-2, 0), square=True)
        t7 = time.time()
        self.log(2, 'vvov sampler time', t7 - t6)
        self.log(0, 'H construction time', t7 - t1)

    @utils.maybe_profile
    def set_T(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.t2T = self.t2.permute((1, 0, 3, 2))
        assert self.t2.is_contiguous()
        self.w1_fT = sum_square(self.t2, dim=(0, 1, 2))
        self.w1_f = self.w1_fV * self.w1_fT
        self.w2_mT = sum_square(self.t2T, dim=(0, 1, 2))
        self.w2_m = self.w2_mV * self.w2_mT
        self.w1_norm = torch.sum(self.w1_f)
        self.w2_norm = torch.sum(self.w2_m)
        self.sampler_kcj = self.constructor_occ(3, self.occ_block_size, True, self.t2.moveaxis(-1, 0), square=True)
        self.sampler_ckb = self.constructor_vir(3, self.vir_block_size, True, self.t2T.moveaxis(-1, 0), square=True)

    def clear_T(self):
        del self.t1, self.t2, self.t2T, self.w1_fT, self.w1_f, self.w2_mT, self.w2_m, self.w1_norm, self.w2_norm, self.sampler_kcj, self.sampler_ckb

    def get_w1(self, ijkabc, tensors=None, square=False):
        if tensors is None:
            eris_vvov = self.eris_vvov
            t2 = self.t2
        else:
            eris_vvov, t2 = tensors
        i, j, k, a, b, c = ijkabc
        return SDDMM(eris_vvov, t2, (a, i, b), (k, c, j), square=square)

    def get_w2(self, ijkabc, tensors=None, square=False):
        if tensors is None:
            eris_vooo = self.eris_vooo
            t2T = self.t2T
        else:
            eris_vooo, t2T = tensors
        i, j, k, a, b, c = ijkabc
        return SDDMM(eris_vooo, t2T, (a, i, j), (c, k, b), square=square)

    def get_v(self, ijkabc, tensors=None):
        if tensors is None:
            eris_oovv, t1 = self.eris_oovv, self.t1
        else:
            eris_oovv, t1 = tensors
        i, j, k, a, b, c = ijkabc
        return eris_oovv[i, a, j, b] * t1[k, c]

    def symm_w1(self, ijkabc, tensors=None, square=False):
        contribution = torch.zeros(ijkabc[0].shape)
        for ijkabc in symm_indices(ijkabc):
            contribution += self.get_w1(ijkabc, tensors=tensors, square=square)
        return contribution

    def symm_w2(self, ijkabc, tensors=None, square=False):
        contribution = torch.zeros(ijkabc[0].shape)
        for ijkabc in symm_indices(ijkabc):
            contribution += self.get_w2(ijkabc, tensors=tensors, square=square)
        return contribution

    def symm_r3_w1(self, ijkabc, tensors=None):
        # Originally 36 terms, merged into 12 calculations by linear combination
        # The original 36 terms can be grouped into 6 terms of [4, -2, -2, 1] (totally 6 * 4 = 24) and 3 terms of [1, 1, -2, -2] (totally 3 * 4 = 12)
        # The [4, -2, -2, 1] four terms can be computed in one evaluation by
        # [4, -2, -2, 1] = kron([2, -1], [2, -1])
        # The [1, -2, -2, 1] four terms can be computed in two evaluation by
        # [1, -2, -2, 1] = kron([1, -1], [1, -1]) * 1.5 - kron([1, 1], [1, 1]) * 0.5
        # Finally the number of evaluations is 6 * 1 + 3 * 2 = 12
        if tensors is None:
            eris_vvov = self.eris_vvov
            t2 = self.t2
        else:
            eris_vvov, t2 = tensors
        I, J, K, A, B, C = ijkabc
        contribution = torch.zeros(len(I))

        V = symmetrize(eris_vvov, 2.0, -1.0, 0, 2)
        T = symmetrize(t2, 2.0, -1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w1((i, j, k, a, b, c), tensors=(V, T)) + self.get_w1((i, k, j, a, c, b), tensors=(V, T))
        del V, T

        V = symmetrize(eris_vvov, 1.0, -1.0, 0, 2)
        T = symmetrize(t2, 1.0, -1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w1((i, j, k, b, c, a), tensors=(V, T)) * 1.5
        del V, T

        V = symmetrize(eris_vvov, 1.0, 1.0, 0, 2)
        T = symmetrize(t2, 1.0, 1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w1((i, j, k, b, c, a), tensors=(V, T)) * -0.5
        del V, T
        return contribution

    def symm_r3_w2(self, ijkabc, tensors=None):
        # Similar treatment as symm_r3_w1
        if tensors is None:
            eris_vooo = self.eris_vooo
            t2T = self.t2T
        else:
            eris_vooo, t2T = tensors
        I, J, K, A, B, C = ijkabc
        contribution = torch.zeros(len(I))

        V = symmetrize(eris_vooo, 2.0, -1.0, 1, 2)
        T = symmetrize(t2T, 2.0, -1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w2((i, j, k, a, b, c), tensors=(V, T)) + self.get_w2((j, i, k, b, a, c), tensors=(V, T))
        del V, T

        V = symmetrize(eris_vooo, 1.0, -1.0, 1, 2)
        T = symmetrize(t2T, 1.0, -1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w2((i, j, k, c, a, b), tensors=(V, T)) * 1.5
        del V, T

        V = symmetrize(eris_vooo, 1.0, 1.0, 1, 2)
        T = symmetrize(t2T, 1.0, 1.0, 0, 2)
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_w2((i, j, k, c, a, b), tensors=(V, T)) * -0.5
        del V, T
        return contribution

    def symm_r3_v(self, ijkabc):
        I, J, K, A, B, C = ijkabc
        t2_tmp1 = self.t2 + self.t2.swapaxes(0, 2).swapaxes(1, 3)
        t2_tmp2 = self.t2.swapaxes(0, 2) + self.t2.swapaxes(1, 3)
        t2_symm = t2_tmp1 - t2_tmp2 * 2
        t2_full = t2_tmp1 * 4 - t2_tmp2 * 2
        contribution = 0
        for i, j, k, a, b, c in [(I, J, K, A, B, C), (J, K, I, B, C, A), (K, I, J, C, A, B)]:
            contribution += self.get_v((i, j, k, a, b, c), tensors=(t2_full, self.t1))
            contribution += self.get_v((j, k, i, a, b, c), tensors=(t2_symm, self.t1))
            contribution += self.get_v((k, i, j, a, b, c), tensors=(t2_symm, self.t1))
        return contribution

    @utils.maybe_profile
    def sample_ijkabc(self, nsamples, random_state):
        n1 = int(self.w1_norm / (self.w1_norm + self.w2_norm) * nsamples)
        n2 = nsamples - n1

        counts_f = sample.make_marginal_nsamples(n1, self.w1_f)
        f = sample.make_marginal_indices(counts_f, n1)
        counts_m = sample.make_marginal_nsamples(n2, self.w2_m)
        m = sample.make_marginal_indices(counts_m, n2)
        #(a1, i1, b1), random_state = self.sampler_aib.sample_indices(f, random_state)
        #(k1, c1, j1), random_state = self.sampler_kcj.sample_indices(f, random_state)
        #(a2, i2, j2), random_state = self.sampler_aij.sample_indices(m, random_state)
        #(c2, k2, b2), random_state = self.sampler_ckb.sample_indices(m, random_state)
        (a1, i1, b1), random_state = sample.sample_indices(self.sampler_aib, f, random_state)
        (k1, c1, j1), random_state = sample.sample_indices(self.sampler_kcj, f, random_state)
        (a2, i2, j2), random_state = sample.sample_indices(self.sampler_aij, m, random_state)
        (c2, k2, b2), random_state = sample.sample_indices(self.sampler_ckb, m, random_state)
        i = torch.concat((i1, i2))
        j = torch.concat((j1, j2))
        k = torch.concat((k1, k2))
        a = torch.concat((a1, a2))
        b = torch.concat((b1, b2))
        c = torch.concat((c1, c2))
        return i, j, k, a, b, c

    def get_eijkabc(self, ijkabc):
        i, j, k, a, b, c = ijkabc
        return self.e_occ[i] + self.e_occ[j] + self.e_occ[k] - self.e_vir[a] - self.e_vir[b] - self.e_vir[c]

    @utils.maybe_profile
    def evaluate_samples(self, ijkabc, symm_sample=True, merge_w=False, merge_v=False):
        if symm_sample:
            w_ijkabc = (self.symm_w1(ijkabc) - self.symm_w2(ijkabc)) / 6
            ps = (self.symm_w1(ijkabc, square=True) + self.symm_w2(ijkabc, square=True)) / 6 / (self.w1_norm + self.w2_norm)
        else:
            w_ijkabc = self.get_w1(ijkabc) - self.get_w2(ijkabc)
            ps = (self.get_w1(ijkabc, square=True) + self.get_w2(ijkabc, square=True)) / (self.w1_norm + self.w2_norm)

        if not (merge_w and merge_v):
            ijkabc_weights_symm_r3 = sum([r3_indices_weights(IJKABC) for IJKABC in symm_indices(ijkabc)], [])
        if merge_w:
            w_ijkabc_symm_r3 = self.symm_r3_w1(ijkabc) - self.symm_r3_w2(ijkabc)
        else:
            w_ijkabc_symm_r3 = sum((self.get_w1(ijkabc_perm) - self.get_w2(ijkabc_perm)) * w for ijkabc_perm, w in ijkabc_weights_symm_r3)
        if merge_v:
            v_ijkabc_symm_r3 = self.symm_r3_v(ijkabc)
        else:
            v_ijkabc_symm_r3 = sum(self.get_v(ijkabc_perm) * w for ijkabc_perm, w in ijkabc_weights_symm_r3)
        samples_w = w_ijkabc_symm_r3 + v_ijkabc_symm_r3 / 2

        d3_ijkabc = self.get_eijkabc(ijkabc)
        weights = (w_ijkabc / d3_ijkabc / ps)
        return samples_w * weights * 2

    def raw_evaluate(self, nsamples, symm_sample=True, merge_w=False, merge_v=False):
        random_state = sample.get_random_state()
        ijkabc = self.sample_ijkabc(nsamples, random_state)
        samples = self.evaluate_samples(ijkabc, symm_sample=symm_sample, merge_w=merge_w, merge_v=merge_v)
        return samples

    def estimate_nsamples(self, target_error):
        def func(nsamples_single):
            samples = self.raw_evaluate(nsamples_single * nrun_estimation, symm_sample=True, merge_w=False, merge_v=False)
            return samples.reshape((nsamples_single, nrun_estimation)).std(dim=0)
        std, _, uncertainty = utils.get_average_until_convergence(func, std_tol, nsamples_test)
        nsamples = int(((std / target_error)**2).item())
        return nsamples, uncertainty

    def evaluate(self, X1, X2, target_error, symm_sample=True, merge_w=False, merge_v=False, retain_T=False):
        t0 = time.time()
        self.set_T(X1, X2)
        t1 = time.time()
        self.log(1, 'T construction time', t1 - t0)

        nsamples, uncertainty = self.estimate_nsamples(target_error)
        t2 = time.time()
        self.log(0, 'Estimated number of samples', nsamples, 'uncertainty', uncertainty)
        self.log(1, 'Estimation time', t2 - t1)
        nsamples = max(nsamples, 10)
        random_state = sample.get_random_state()
        ijkabc = self.sample_ijkabc(nsamples, random_state)
        t3 = time.time()
        self.log(1, 'Sampling time', t3 - t2)
        samples = self.evaluate_samples(ijkabc, symm_sample=symm_sample, merge_w=merge_w, merge_v=merge_v)
        t4 = time.time()
        self.log(1, 'Evaluation time', t4 - t3)
        self.log(0, 'Total time', t4 - t0)
        if not retain_T:
            self.clear_T()
        return samples
