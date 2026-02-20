import time
import functools
import builtins
import typing as tp
import torch
from . import la, sample, stc, CCSD, utils, utils_pyscf, diis, pt
from torch._subclasses.fake_tensor import FakeTensorMode

if not hasattr(builtins, 'profile'):
    profile = lambda x: x

jit_estimation = True
use_alias_for_X2vvoo = True
make_X2vvoo_contiguous = False

test_nsamples = 10000
std_tol_X1 = 0.05
std_tol_X2 = 0.02
nrun_estimation = 8
compile_threshold = 0.0


def build_index_sizes(nocc, nvir, naux):
    return {'i': nocc, 'j': nocc, 'k': nocc, 'l': nocc, 
            'a': nvir, 'b': nvir, 'c': nvir, 'd': nvir,
            'x': naux }


def get_ovov_mix(X):
    C1 = torch.tensor(2.0)
    C2 = torch.tensor(-1.0)
    X_mix = la.linear_combination((X, X.swapaxes(0, 2)), (C1, C2))
    return X_mix


def transform_X1(T, Uocc, Uvir):
    T = utils.pop_reference_data(T)
    T = torch.einsum('Aa,ia->Ai', Uvir, T)
    T = torch.einsum('Ii,Ai->IA', Uocc, T)
    return T


def transform_X2(T, Uocc, Uvir):
    T = utils.pop_reference_data(T)
    T = torch.einsum('Bb,iajb->Biaj', Uvir, T)
    T = torch.einsum('Jj,Biaj->JBia', Uocc, T)
    T = torch.einsum('Aa,JBia->AJBi', Uvir, T)
    T = torch.einsum('Ii,AJBi->IAJB', Uocc, T)
    return T


def assign_nsamples(stds, target_error, weights=None):
    # minimize sum_i Ni*wi while keeping sum_i si^2/Ni = target_error^2
    # Ni = si / sqrt(wi) * sum_i (si * sqrt(wi)) / target_error^2
    if weights is None:
        weights = torch.ones_like(stds)
    tot_std = torch.sum(stds * torch.sqrt(weights))
    nsamples_all = stds / torch.sqrt(weights) * tot_std / target_error**2
    return nsamples_all


@functools.partial(torch.compile, dynamic=True)
def X2_apply_denominator_canonical_static(X, eov, inv=False):
    eiajb = eov[:, :, None, None] + eov[None, None, :, :]
    XT = torch.permute(X, (2, 3, 0, 1))
    if inv:
        return (X + XT) * eiajb * 0.5
    else:
        return (X + XT) / eiajb * 0.5


@functools.partial(torch.compile, dynamic=True)
def get_X_full_static(X1A, X1B, X2):
    return X2 + torch.einsum('ia,jb->iajb', X1A, X1B)


@functools.partial(torch.compile, dynamic=True)
def get_Eij_static(X_full, Hovov, symmetrize=True):
    if symmetrize:
        return torch.sum(X_full * Hovov, dim=(1, 3)) * 2 - torch.sum(X_full * Hovov.swapaxes(0, 2), dim=(1, 3))
    else:
        return torch.sum(X_full * Hovov, dim=(1, 3))
    

@functools.partial(torch.compile, dynamic=True)
def get_X1_energy_static_fast(X1A, X1B, Hovov):
    return torch.einsum('iajb,ia,jb->', Hovov, X1A, X1B) * 2 - torch.einsum('iajb,ib,ja->', Hovov, X1A, X1B)
    

@functools.partial(torch.compile, dynamic=True)
def flatten_X(X1A, X1B, X2):
    nocc, nvir, _, _ = X2.shape
    idx1, idx2 = torch.triu_indices(nocc, nocc)
    X2_triu_flat = X2[idx1, :, idx2, :]
    return torch.cat([X1A.reshape(-1), X1B.reshape(-1), X2_triu_flat.reshape(-1)])


@functools.partial(torch.compile, dynamic=True)
def unflatten_X(X_flat, nocc, nvir):
    nocc_triu = nocc * (nocc + 1) // 2
    size_X1 = nocc * nvir
    size_X2 = nocc_triu * nvir * nvir
    assert X_flat.numel() == size_X1 * 2 + size_X2
    X1A_flat = X_flat[0:size_X1]
    X1B_flat = X_flat[size_X1:size_X1 * 2]
    X2_triu_flat = X_flat[size_X1 * 2:]
    X1A = X1A_flat.reshape((nocc, nvir))
    X1B = X1B_flat.reshape((nocc, nvir))
    X2_triu = X2_triu_flat.reshape((nocc_triu, nvir, nvir))
    X2 = torch.zeros((nocc, nvir, nocc, nvir))
    idx1, idx2 = torch.triu_indices(nocc, nocc)
    X2[idx1, :, idx2, :] = X2_triu
    X2[idx2, :, idx1, :] = X2_triu.swapaxes(1, 2)
    return X1A, X1B, X2


symm_abcd = stc.get_symmetry(4, [])
symm_ab_ab = stc.get_symmetry(4, [(2, 3, 0, 1)])
symm_aa_bb = stc.get_symmetry(4, [(1, 0, 3, 2)])
symm_abab = stc.get_symmetry(4, [(2, 1, 0, 3), (0, 3, 2, 1)])
symm_aabb = stc.get_symmetry(4, [(1, 0, 2, 3), (0, 1, 3, 2)])
symm_aaaa = stc.get_symmetry(4, [(2, 3, 0, 1), (1, 0, 2, 3), (0, 1, 3, 2)])
symm_aabc = stc.get_symmetry(4, [(1, 0, 2, 3)])
symm_abcc = stc.get_symmetry(4, [(0, 1, 3, 2)])
symm_abc = stc.get_symmetry(3, [])
symm_aab = stc.get_symmetry(3, [(1, 0, 2)])
symm_aa = stc.get_symmetry(2, [(1, 0)])
symm_ab = stc.get_symmetry(2, [])


def maybe_pad_tensor(tensor, full_shape, value=None):
    if tensor.shape == full_shape:
        return tensor
    assert tensor.ndim == len(full_shape)
    npad = [(0, full - cur) for cur, full in zip(tensor.shape, full_shape)]
    return torch.nn.functional.pad(tensor, [p for pad in reversed(npad) for p in pad], value=value)


class StochasticCC(utils.Logging):

    def __init__(self, rhf, C_local, target_error, block_size=None, padding=(False, False, False), weights_out_power=0, Uaux=None, canonical_denominator=True, minimal_stc=True, linear=False, singles=True, diis_space=None, diis_shift=None, jit=True, dynamic=True, verbose=0):
        self.rhf = rhf
        self.pbc = hasattr(self.rhf, 'cell')
        self.mol_or_cell = self.rhf.cell if self.pbc else self.rhf.mol
        self.nao = self.mol_or_cell.nao
        self.ncore = self.nao - C_local.shape[1]
        self.nocc = self.mol_or_cell.nelectron // 2
        self.nvir = self.nao - self.nocc
        self.naux = rhf.with_df.get_naoaux()
        self.o = slice(self.ncore, self.nocc)
        self.v = slice(self.nocc, None)
        self.nocc = self.nocc - self.ncore
        self.verbose = verbose
        self.log(0, 'nocc', self.nocc)
        self.log(0, 'nvir', self.nvir)
        self.log(0, 'naux', self.naux)
        self.df_virtual = Uaux is not None
        self.block_size = block_size
        self.pad_occ, self.pad_vir, self.pad_aux = padding
        self.nocc_full = utils.get_nbatches(self.nocc, self.block_size) * self.block_size if self.pad_occ else self.nocc
        self.nvir_full = utils.get_nbatches(self.nvir, self.block_size) * self.block_size if self.pad_vir else self.nvir
        self.naux_full = utils.get_nbatches(self.naux, self.block_size) * self.block_size if self.pad_aux else self.naux
        self.indices_to_pad = (CCSD.indices_occ if self.pad_occ else '') + (CCSD.indices_vir if self.pad_vir else '') + (CCSD.indices_aux if self.pad_aux else '')
        if self.pad_occ:
            self.log(0, 'pad nocc to', self.nocc_full)
        if self.pad_vir:
            self.log(0, 'pad nvir to', self.nvir_full)
        if self.pad_aux:
            self.log(0, 'pad naux to', self.naux_full)
        if self.block_size is not None:
            assert min(self.nocc_full, self.nvir_full, self.naux_full) * 2 >= self.block_size, "block size is too large"
            
        self.target_error = target_error
        self.mask = None
        self.singles = singles
        self.jit = jit
        self.jit_estimation = jit and jit_estimation
        self.dynamic = dynamic

        C = torch.from_numpy(self.rhf.mo_coeff)
        self.Cocc = C[:, self.o]
        self.Cvir = C[:, self.v]
        self.C = torch.concat([self.Cocc, self.Cvir], dim=1)
        C_local = torch.from_numpy(C_local)
        self.Cocc_local = C_local[:, 0:self.nocc]
        self.Cvir_local = C_local[:, self.nocc:]
        self.Uaux = None if Uaux is None else torch.from_numpy(Uaux)

        self.Cocc = maybe_pad_tensor(self.Cocc, (self.nao, self.nocc_full))
        self.Cvir = maybe_pad_tensor(self.Cvir, (self.nao, self.nvir_full))
        self.Cocc_local = maybe_pad_tensor(self.Cocc_local, (self.nao, self.nocc_full))
        self.Cvir_local = maybe_pad_tensor(self.Cvir_local, (self.nao, self.nvir_full))
        self.C_local = torch.concat([self.Cocc_local, self.Cvir_local], dim=1)
        if self.Uaux is not None:
            self.Uaux = maybe_pad_tensor(self.Uaux, (self.naux_full, self.naux_full))

        self.canonical_denominator = canonical_denominator
        self.minimal_stc = minimal_stc
        self.add_F = (not canonical_denominator)
        self.linear = linear
        self.intermediate_tensors_list = CCSD.get_intermediate_tensors(self.minimal_stc, self.linear)
        self.X1_shape = (self.nocc_full, self.nvir_full)
        self.X2_shape = (self.nocc_full, self.nvir_full, self.nocc_full, self.nvir_full)
        self.diagrams_X1 = CCSD.get_CCSD_X1(self.minimal_stc, self.df_virtual, self.linear) if self.singles else []
        self.diagrams_X2 = CCSD.get_CCSD_X2(self.minimal_stc, self.df_virtual, self.linear)
        self.intermediate_tensors_list = CCSD.screen_unused_intermediates(self.intermediate_tensors_list, self.diagrams_X1 + self.diagrams_X2)
        self.updated_symbols = [symbol for symbol, _ in self.intermediate_tensors_list] + ['X1', 'X2', 'X2_symm', 'X2_vvoo']

        self.build_symm_dict()
        self.value_dict = {}

        S = self.mol_or_cell.pbc_intor('int1e_ovlp') if self.pbc else self.mol_or_cell.intor('int1e_ovlp')
        self.S = torch.from_numpy(S)
        self.Uocc = self.Cocc_local.T @ self.S @ self.Cocc
        self.Uvir = self.Cvir_local.T @ self.S @ self.Cvir

        self.build_denominator()

        self.diis_space = diis_space
        self.diis_shift = diis_shift
        self.vec_size = self.nocc_full * (self.nocc_full + 1) // 2 * self.nvir_full**2 + self.nocc_full * self.nvir_full * 2
        self.err_size = self.nocc_full**2
        self.initialize_diis()
        self.weights_out_power = weights_out_power
        self.value_cache = stc.ValueCache(self.value_dict, self.weights_out_power)
        self.nsamples_history = []

        self.min_nsamples_tot = None
        self.max_nsamples_tot = None
        self.weights_out_E = True

    def initialize_diis(self):
        if self.diis_space is None:
            self.diis = None
        else:
            shift = None if self.diis_shift is None else self.target_error**2 * self.diis_shift * (self.diis_space - 1)
            self.diis = diis.DIIS(self.vec_size, self.err_size, self.diis_space, shift=shift, verbose=(self.verbose >= 1))

    def initialize(self):
        begin = time.time()
        this_begin = time.time()
        R_ao = utils_pyscf.get_DF_tensor(self.rhf.with_df, transpose=True)
        R_ao = torch.from_numpy(R_ao)
        R_ao = maybe_pad_tensor(R_ao, (self.nao, self.nao, self.naux_full))
        this_end = time.time()
        self.log(1, 'constructing Rao time', this_end - this_begin)

        this_begin = time.time()
        R_local = la.transform_from_left(R_ao, [self.C_local.T, self.C_local.T, self.Uaux.T if (self.Uaux is not None) else None])
        this_end = time.time()
        self.log(1, 'constructing Rlocal time', this_end - this_begin)
        del R_ao
        self.add_R(R_local)
        del R_local

        this_begin = time.time()
        self.build_H()
        this_end = time.time()
        self.log(1, 'constructing H time', this_end - this_begin)

        self.Hovov = self.value_dict['Hovov']

        this_begin = time.time()
        self.T_local_antisym = None
        T_local = self.get_T_local()
        self.T_local_antisym = get_ovov_mix(T_local)

        if self.weights_out_E:
            self.value_dict['O2'] = self.T_local_antisym
        else:
            eov_local = self.evir_local[None, :] - self.eocc_local[:, None]
            eovov_local = eov_local[:, :, None, None] + eov_local[None, None, :, :]
            self.value_dict['O2'] = get_ovov_mix(1 / eovov_local)
        self.value_dict['O1'] = torch.sqrt(torch.sum(torch.abs(self.value_dict['O2']), axis=(2, 3)))
        self.build_intermediate_tensors_fake()
        diagrams = self.screen_diagrams(self.diagrams_X2, stochastic=True)
        graphs = self.get_einsum_graphs(diagrams, 'O2', evaluate=False)
        keys = set.union(*[g.get_keys_recursively() for g in graphs])
        for key in keys:
            if isinstance(key, tuple):
                self.value_cache.set_weights_out(key)
        self.clear_value_dict()
        with FakeTensorMode():
            self.value_dict['O2'] = torch.zeros(self.X2_shape)

        #self.value_dict['O1'] = torch.ones((self.nocc, self.nvir))
        this_end = time.time()
        self.log(1, 'constructing weights_out time', this_end - this_begin)

        end = time.time()
        self.log(0, 'StochasticCC initialization time', end - begin)

    def clear(self):
        del self.value_dict, self.value_cache, self.Hovov, self.T_local_antisym, self.diis

    def build_symm_dict(self):
        self.symm_dict = {
            'X1': symm_ab,
            'X2': symm_ab_ab,
            'X2_symm': symm_ab_ab,
            'X2_vvoo': symm_aa_bb,
            'Roo': symm_aab,
            'Rov': symm_abc,
            'Rvv': symm_aab,
            'Loo': symm_ab,
            'Lvv': symm_ab,
            'Hoooo': symm_aaaa,
            'Hvvvv': symm_aaaa,
            'Hovov': symm_ab_ab,
            'Hoovv': symm_abab if CCSD.reorder_Hoovv else symm_aabb,
            'Hovoo': symm_abcc,
            'Hovvv': symm_abcc,
            'O1': symm_ab,
            'O2': symm_ab_ab,
        }

    def build_denominator(self):
        e = torch.from_numpy(self.rhf.mo_energy.copy())
        self.eocc_canonical = e[self.o]
        self.evir_canonical = e[self.v]
        self.eocc_canonical = maybe_pad_tensor(self.eocc_canonical, (self.nocc_full, ), value=-torch.inf)
        self.evir_canonical = maybe_pad_tensor(self.evir_canonical, (self.nvir_full, ), value=torch.inf)
        self.F_ao = torch.from_numpy(self.rhf.get_fock())
        self.Focc_local = self.Cocc_local.T @ self.F_ao @ self.Cocc_local
        self.Fvir_local = self.Cvir_local.T @ self.F_ao @ self.Cvir_local
        self.eocc_local = torch.diag(self.Focc_local).clone(memory_format=torch.contiguous_format)
        self.evir_local = torch.diag(self.Fvir_local).clone(memory_format=torch.contiguous_format)
        self.eocc_local[torch.arange(self.nocc, self.nocc_full)] = -torch.inf
        self.evir_local[torch.arange(self.nvir, self.nvir_full)] = torch.inf
        if self.canonical_denominator:
            self.eocc = self.eocc_canonical.clone(memory_format=torch.contiguous_format)
            self.evir = self.evir_canonical.clone(memory_format=torch.contiguous_format)
            self.Focc = None
            self.Fvir = None
        else:
            self.eocc = self.eocc_local.clone(memory_format=torch.contiguous_format)
            self.evir = self.evir_local.clone(memory_format=torch.contiguous_format)
            self.Focc = self.Focc_local.clone(memory_format=torch.contiguous_format)
            self.Fvir = self.Fvir_local.clone(memory_format=torch.contiguous_format)
            self.Focc[torch.arange(self.nocc_full), torch.arange(self.nocc_full)] = 0.0
            self.Fvir[torch.arange(self.nvir_full), torch.arange(self.nvir_full)] = 0.0

        #self.eocc_canonical = maybe_pad_tensor(self.eocc_canonical, (self.nocc_full, ), value=-torch.inf)
        #self.evir_canonical = maybe_pad_tensor(self.evir_canonical, (self.nvir_full, ), value=torch.inf)
        #self.eocc = maybe_pad_tensor(self.eocc, (self.nocc_full, ), value=-torch.inf)
        #self.evir = maybe_pad_tensor(self.evir, (self.nvir_full, ), value=torch.inf)
        #if not self.canonical_denominator:
        #    self.Focc = maybe_pad_tensor(self.Focc, (self.nocc_full, self.nocc_full))
        #    self.Fvir = maybe_pad_tensor(self.Fvir, (self.nvir_full, self.nvir_full))

        self.value_dict['Focc'] = self.Focc
        self.value_dict['Fvir'] = self.Fvir
        if self.pbc:
            import pyscf.pbc.tools
            self.madelung = torch.tensor(pyscf.pbc.tools.madelung(self.rhf.cell, self.rhf.kpt))
        else:
            self.madelung = None
        self.value_dict['madelung'] = self.madelung
        self.eov = self.eocc[:, None] - self.evir[None, :]

    def add_R(self, R):
        o = slice(0, self.nocc_full)
        v = slice(self.nocc_full, self.nocc_full + self.nvir_full)
        self.value_dict['Roo'] = R[o, o, :].clone(memory_format=torch.contiguous_format)
        self.value_dict['Rov'] = R[o, v, :].clone(memory_format=torch.contiguous_format)
        self.value_dict['Rvo'] = R[v, o, :].clone(memory_format=torch.contiguous_format)
        self.value_dict['Rvv'] = R[v, v, :].clone(memory_format=torch.contiguous_format)
        self.value_dict['RovT'] = torch.moveaxis(self.value_dict['Rov'], 2, 0).clone(memory_format=torch.contiguous_format)

    def build_H(self):
        begin = time.time()
        self.value_dict['Hoooo'] = self.PQ2H(self.value_dict['Roo'], self.value_dict['Roo'], name='Hoooo')
        self.value_dict['Hovov'] = self.PQ2H(self.value_dict['Rov'], self.value_dict['RovT'], transpose_Q=True, name='Hovov')
        self.value_dict['Hovoo'] = self.PQ2H(self.value_dict['Rov'], self.value_dict['Roo'], name='Hovoo')
        self.value_dict['Hoovv'] = self.PQ2H(self.value_dict['Roo'], self.value_dict['Rvv'], name='Hoovv')
        if CCSD.reorder_Hoovv:
            self.value_dict['Hoovv'] = self.value_dict['Hoovv'].swapaxes(1, 2).clone(memory_format=torch.contiguous_format)
        if not self.df_virtual:
            self.value_dict['Hovvv'] = self.PQ2H(self.value_dict['Rov'], self.value_dict['Rvv'], name='Hovvv')
            self.value_dict['Hvvvv'] = self.PQ2H(self.value_dict['Rvv'], self.value_dict['Rvv'], name='Hvvvv')
        end = time.time()
        self.log(1, 'H construction time', end - begin)

    @utils.reference_decorate
    def PQ2H(self, P, Q, transpose_Q=False, name=None):
        assert P.is_contiguous()
        assert Q.is_contiguous()
        x = P.shape[-1]
        nP = P.shape[0] * P.shape[1]
        t1 = time.time()
        if transpose_Q:
            assert Q.shape[0] == x
            nQ = Q.shape[1] * Q.shape[2]
            result = torch.empty(P.shape[:-1] + Q.shape[1:])
            torch.matmul(P.view((-1, x)), Q.view((x, -1)), out=result.view((nP, nQ)))
            #result = torch.einsum('ikx,xjl->ikjl', P, Q)
        else:
            assert Q.shape[-1] == x
            nQ = Q.shape[0] * Q.shape[1]
            result = torch.empty(P.shape[:-1] + Q.shape[:-1])
            torch.matmul(P.view((-1, x)), Q.view((-1, x)).T, out=result.view((nP, nQ)))
            #result = torch.einsum('ikx,jlx->ikjl', P, Q)
        #assert result.is_contiguous()
        t2 = time.time()
        self.log(2, f'PQ2H {name} time', t2 - t1)
        assert result._base is None
        return result
    
    def construct_sampler(self, M1_tuple, M2_tuple):
        M1, t1, M1_indices = M1_tuple
        M2, t2, M2_indices = M2_tuple
        symbols = set(stc.get_symbols(t1)).union(set(stc.get_symbols(t2)))
        is_fixed = len(set(self.updated_symbols).intersection(symbols)) == 0
        out_dim = len(set(M1_indices).intersection(set(M2_indices)))
        assert M1_indices[-out_dim:] == M2_indices[-out_dim:]
        indices_all = set(M1_indices).union(set(M2_indices))
        nfree1 = M1.ndim - out_dim
        nfree2 = M2.ndim - out_dim
        ntot = nfree1 + nfree2 + out_dim
        is_X2_vvoo = stc.get_symbols(t1) == ('X2_vvoo', )
        score = sum((2 if idx in CCSD.indices_occ else 3) for idx in indices_all)
        assert ntot <= 4
        full_sampler = score <= 8 or (score == 9 and is_fixed) or (use_alias_for_X2vvoo and is_X2_vvoo) # score == 9 include ooov and vvx
        if full_sampler or (self.block_size is None):
            sampler = sample.BatchedAliasSampler(out_dim, M1, M2)
        else:
            index_last = M1_indices[-1]
            constructor = sample.AlignedBatchedBlockSampler if index_last in self.indices_to_pad else sample.BatchedBlockSampler
            dim_last = M1.shape[-out_dim:-1].numel()
            nblocks_last = (dim_last - 1) // self.block_size + 1
            nblocks = M1.shape[-out_dim:-1].numel() * nblocks_last
            alias = nblocks > 4
            sampler = constructor(out_dim, self.block_size, alias, M1, M2)
        return sampler

    def screen_diagrams(self, diagrams, stochastic=None):
        def is_none(diagram):
            return any(self.value_dict[symbol] is None for symbol in diagram.symbols)

        diagrams = [diagram for diagram in diagrams if not is_none(diagram)]
        if stochastic is not None:
            #diagrams = [diagram for diagram in diagrams if diagram.is_stochastic == stochastic and diagram.expr == 'acx,bdx,icjd->iajb']
            diagrams = [diagram for diagram in diagrams if diagram.is_stochastic == stochastic]
        return diagrams

    def get_einsum_graphs(self, diagrams, out_symbol, evaluate=False) -> tp.List[stc.EinsumGraph]:
        graphs = []
        for diagram in diagrams:
            graph = stc.EinsumGraph.from_diagram(diagram, self.symm_dict, out_symbol, verbose=self.verbose)
            if evaluate:
                graph.evaluate_graph(self.construct_sampler, self.value_cache)
            graphs.append(graph)
        return graphs

    @utils.reference_decorate
    def get_T_local(self):
        if self.T_local_antisym is not None:
            T = self.T_local_antisym
            T2 = T.swapaxes(0, 2)
            C1 = torch.tensor(2.0/3.0)
            C2 = torch.tensor(1.0/3.0)
            return la.linear_combination((T, T2), (C1, C2))

        begin = time.time()
        if self.canonical_denominator:
            Rov_canonical = la.transform_from_left(self.value_dict['Rov'], [self.Uocc.T, self.Uvir.T, None]).contiguous()
            RovT_canonical = torch.moveaxis(Rov_canonical, 2, 0).contiguous()
            self.log(2, 'local Rov to canonical time', time.time() - begin)
            result = self.PQ2H(Rov_canonical, RovT_canonical, transpose_Q=True, name='ovov', wrap_ref=True)
            result = self.X2_apply_denominator_canonical(result, wrap_ref=True)
            result = self.X2_canonical_to_local(result, wrap_ref=True)
        else:
            result = self.X2_apply_denominator_canonical(self.Hovov, wrap_ref=True)
        end = time.time()
        self.log(1, 'local T construction time', end - begin)
        return result.pop()

    @utils.reference_decorate
    def get_X1_guess(self):
        return None

    @utils.reference_decorate
    def get_X2_guess(self):
        return self.get_T_local()

    @utils.reference_decorate
    def X1_local_to_canonical(self, X1):
        return la.transform_from_left(X1, [self.Uocc.T, self.Uvir.T])

    @utils.reference_decorate
    def X1_canonical_to_local(self, X1):
        return la.transform_from_left(X1, [self.Uocc, self.Uvir])

    @utils.reference_decorate
    def X1_apply_denominator_local(self, X1, inv=False):
        if self.canonical_denominator:
            X1 = self.X1_local_to_canonical(X1)
            if inv:
                X1 = X1 * self.eov
            else:
                X1 = X1 / self.eov
            X1 = self.X1_canonical_to_local(X1)
        else:
            X1 = utils.pop_reference_data(X1)
            if inv:
                X1 = X1 * self.eov
            else:
                X1 = X1 / self.eov
        return X1

    @utils.reference_decorate
    def X2_canonical_to_local(self, X2):
        begin = time.time()
        Uocc, Uvir = self.Uocc, self.Uvir
        #result = transform_X2(X2, Uocc, Uvir)
        result = la.transform_from_left(X2, [Uocc, Uvir, Uocc, Uvir])
        end = time.time()
        self.log(2, 'canonical to local time', end - begin)
        return result

    @utils.reference_decorate
    def X2_local_to_canonical(self, X2):
        begin = time.time()
        Uocc, Uvir = self.Uocc.T.contiguous(), self.Uvir.T.contiguous()
        #result = transform_X2(X2, Uocc, Uvir)
        result = la.transform_from_left(X2, [Uocc, Uvir, Uocc, Uvir])
        end = time.time()
        self.log(2, 'local to canonical time', end - begin)
        return result

    @utils.reference_decorate
    def X2_apply_denominator_canonical(self, X2, inv=False):
        X2 = utils.pop_reference_data(X2)
        begin = time.time()
        result = X2_apply_denominator_canonical_static(X2, self.eov, inv=inv)
        end = time.time()
        self.log(2, 'apply demonimator time', end - begin)
        return result

    @utils.reference_decorate
    def X2_apply_denominator_local(self, X2, inv=False):
        if self.canonical_denominator:
            X2 = self.X2_local_to_canonical(X2, wrap_ref=True)
            X2 = self.X2_apply_denominator_canonical(X2, inv=inv, wrap_ref=True)
            X2 = self.X2_canonical_to_local(X2, wrap_ref=True)
        else:
            X2 = self.X2_apply_denominator_canonical(X2, inv=inv, wrap_ref=True)
        #X2 = X2.contiguous()
        return X2.pop()

    def get_X_full(self, X_tuple):
        X_tuple = utils.get_reference_data(X_tuple)
        X1A, X1B, X2 = X_tuple
        if X1A is None:
            X1A = torch.zeros(self.X1_shape)
        if X1B is None:
            X1B = torch.zeros(self.X1_shape)
        return get_X_full_static(X1A, X1B, X2)

    def get_Eij(self, X_tuple, raw=False):
        X_full = self.get_X_full(X_tuple)
        if raw:
            Eij = get_Eij_static(X_full, self.T_local_antisym, symmetrize=False)
        else:
            Eij = get_Eij_static(X_full, self.Hovov, symmetrize=True)
        return Eij

    def get_energy(self, X_tuple, raw=False):
        return torch.sum(self.get_Eij(X_tuple, raw=raw))

    def build_intermediate_tensors(self, X1, X2, fake=False):
        self.value_dict['X1'] = X1
        self.value_dict['X2'] = X2
        if fake:
            self.value_dict['X2_symm'] = X2 * 2 - X2.swapaxes(0, 2)
        else:
            self.value_dict['X2_symm'] = get_ovov_mix(X2)
        self.value_dict['X2_vvoo'] = X2.permute(1, 3, 0, 2)
        einsum_func = torch.einsum if fake else la.fast_einsum
        if make_X2vvoo_contiguous:
            self.value_dict['X2_vvoo'] = self.value_dict['X2_vvoo'].contiguous()
        for symbol, diagrams in self.intermediate_tensors_list:
            result = None
            begin = time.time()
            for diagram in diagrams:
                tensors = [self.value_dict[key] for key in diagram.symbols]
                if any(t is None for t in tensors):
                    continue
                if result is None:
                    result = einsum_func(diagram.expr, *tensors) * diagram.coeff
                else:
                    result += einsum_func(diagram.expr, *tensors) * diagram.coeff
            end = time.time()
            if (not fake) and (result is not None):
                self.log(3, f'intermediate {symbol} time {end - begin:.6f}')
            self.value_dict[symbol] = result
            if (result is not None) and symbol not in self.symm_dict:
                self.symm_dict[symbol] = stc.get_symmetry(result.ndim, [])

    def build_intermediate_tensors_fake(self):
        with FakeTensorMode(allow_non_fake_inputs=True):
            X1 = torch.zeros(self.X1_shape) if self.singles else None
            X2 = torch.zeros(self.X2_shape)
            self.build_intermediate_tensors(X1, X2, fake=True)

    def clear_value_cache(self):
        self.value_cache.clear_symbols(self.updated_symbols)

    def clear_value_dict(self):
        for symbol in ['X1', 'X2', 'X2_symm', 'X2_vvoo']:
            if symbol in self.value_dict:
                del self.value_dict[symbol]
        for symbol, _ in self.intermediate_tensors_list:
            if symbol in self.value_dict:
                del self.value_dict[symbol]

    def get_X1_stds(self):
        diagrams_X1_exact = self.screen_diagrams(self.diagrams_X1, stochastic=False)
        diagrams_X1_stochastic = self.screen_diagrams(self.diagrams_X1, stochastic=True)
        if len(diagrams_X1_stochastic) == 0:
            return torch.zeros((0, ))

        def func(nsamples):
            graphs_X1 = self.get_einsum_graphs(diagrams_X1_stochastic, 'O1', evaluate=True)
            X1_update = torch.zeros(self.X1_shape)
            for diagram in diagrams_X1_exact:
                self.evaluate_exact(diagram, out=X1_update)
            for graph in graphs_X1:
                graph.evaluate_tensor_contraction(nsamples, X1_update, jit=self.jit_estimation, dynamic=self.dynamic)
            X1_update = self.X1_apply_denominator_local(X1_update)
            X1_update_transform = torch.einsum('iajb,ia->jb', self.Hovov, X1_update) * 2 - torch.einsum('iajb,ja->ib', self.Hovov, X1_update)
            X1_update_transform = self.X1_apply_denominator_local(X1_update_transform)
            variances = []
            for graph in graphs_X1:
                _, Estd, _ = graph.estimate_stats(nsamples, nrun=nrun_estimation, weights_out=X1_update_transform, jit=self.jit_estimation, dynamic=self.dynamic)
                variances.append(Estd**2 * 2)
            return torch.stack(variances)
        
        t1 = time.time()
        var, nsamples, uncertainty = utils.get_average_until_convergence(func, std_tol_X1, test_nsamples)
        t2 = time.time()
        self.log(3, 'X1 time', f'{t2 - t1:.6f}', f'nsamples {nrun_estimation * nsamples:8d}', 'uncertainty', f'{uncertainty:.6f}')
        return torch.sqrt(var)

    def get_X2_stds(self):
        diagrams_X2_stochastic = self.screen_diagrams(self.diagrams_X2, stochastic=True)
        graphs_X2 = self.get_einsum_graphs(diagrams_X2_stochastic, 'O2', evaluate=True)

        def func(nsamples):
            _, Estd, _ = graph.estimate_stats(nsamples, nrun=nrun_estimation, weights_out=self.T_local_antisym, jit=self.jit_estimation, dynamic=self.dynamic)
            return Estd**2

        max_length = max(len(str(graph)) for graph in graphs_X2)
        stds = []
        for diagram, graph in zip(diagrams_X2_stochastic, graphs_X2):
            std_tol = std_tol_X1 if ('X1' in diagram.symbols) else std_tol_X2
            t1 = time.time()
            var, nsamples, uncertainty = utils.get_average_until_convergence(func, std_tol, test_nsamples)
            t2 = time.time()
            graph_name = str(graph).ljust(max_length)
            self.log(3, graph_name, 'time', f'{t2 - t1:.6f}', 'nsamples', nrun_estimation * nsamples, 'uncertainty', f'{uncertainty:.6f}')
            stds.append(torch.sqrt(var))
        return torch.stack(stds)

    def get_X_shape(self, symbol):
        return {'X1': self.X1_shape, 'X2': self.X2_shape}[symbol]

    def get_X_diagrams(self, symbol):
        return {'X1': self.diagrams_X1, 'X2': self.diagrams_X2}[symbol]

    def get_out_symbol(self, symbol):
        return {'X1': 'O1', 'X2': 'O2'}[symbol]

    def get_X_guess(self, symbol):
        return {'X1': self.get_X1_guess, 'X2': self.get_X2_guess}[symbol]()

    def X_apply_denominator_local(self, symbol, X):
        return {'X1': self.X1_apply_denominator_local, 'X2': self.X2_apply_denominator_local}[symbol](X)

    def get_X_stds(self, symbol):
        return {'X1': self.get_X1_stds, 'X2': self.get_X2_stds}[symbol]()

    @utils.reference_decorate
    def get_X_tuple_guess(self):
        X1_guess = self.get_X1_guess()
        X2_guess = self.get_X2_guess()
        return X1_guess, X1_guess, X2_guess

    def get_fake_X1(self):
        Roo = self.value_dict['Roo']
        Rov = self.value_dict['Rov']
        Rvv = self.value_dict['Rvv']
        return torch.einsum('acx,icx->ia', Rvv, Rov) - torch.einsum('kix,kax->ia', Roo, Rov)

    def check_correctness(self, nsamples, X1=None, X2=None):
        if X1 is None:
            X1 = self.get_fake_X1() if self.singles else None
        if X2 is None:
            X2 = self.get_T_local()
        self.build_intermediate_tensors(X1, X2)
        diagrams_X1_stochastic = self.screen_diagrams(self.diagrams_X1, stochastic=True)
        diagrams_X2_stochastic = self.screen_diagrams(self.diagrams_X2, stochastic=True)
        einsum_graphs_X1 = self.get_einsum_graphs(diagrams_X1_stochastic, 'O1', evaluate=True)
        einsum_graphs_X2 = self.get_einsum_graphs(diagrams_X2_stochastic, 'O2', evaluate=True)
        if self.singles:
            diagrams = diagrams_X1_stochastic + diagrams_X2_stochastic
            einsum_graphs = einsum_graphs_X1 + einsum_graphs_X2
        else:
            diagrams = diagrams_X2_stochastic
            einsum_graphs = einsum_graphs_X2
        for diagram, graph in zip(diagrams, einsum_graphs):
            X_update = torch.zeros_like(self.value_dict[graph.out_symbol])
            graph.evaluate_tensor_contraction(nsamples, X_update, jit=self.jit, dynamic=self.dynamic)
            _, _, Tstd = graph.estimate_stats(nsamples, jit=self.jit, dynamic=self.dynamic)
            X_ref = self.evaluate_exact(diagram)
            err = torch.linalg.norm(X_update - X_ref)
            norm = torch.linalg.norm(X_ref)
            relative_err = err / norm
            relative_std = Tstd / torch.sqrt(torch.tensor(nsamples)) / norm
            diff = (relative_err - relative_std) / relative_std
            print(f'{str(graph):50s}    Terr {relative_err:10.6f}    Tstd {relative_std:10.6f}    diff {diff:10.6f}')
        self.clear_value_dict()
        self.clear_value_cache()

    def check_correctness_full(self, nsamples, nrun, X1=None, X2=None):
        if X1 is None:
            X1 = self.get_fake_X1() if self.singles else None
        if X2 is None:
            X2 = self.get_T_local()
        self.build_intermediate_tensors(X1, X2)
        diagrams_X1_stochastic = self.screen_diagrams(self.diagrams_X1, stochastic=True)
        diagrams_X2_stochastic = self.screen_diagrams(self.diagrams_X2, stochastic=True)
        einsum_graphs_X1 = self.get_einsum_graphs(diagrams_X1_stochastic, 'O1', evaluate=True)
        einsum_graphs_X2 = self.get_einsum_graphs(diagrams_X2_stochastic, 'O2', evaluate=True)
        if self.singles:
            diagrams = diagrams_X1_stochastic + diagrams_X2_stochastic
            einsum_graphs = einsum_graphs_X1 + einsum_graphs_X2
        else:
            diagrams = diagrams_X2_stochastic
            einsum_graphs = einsum_graphs_X2
        for diagram, graph in zip(diagrams, einsum_graphs):
            Eerrs = []
            Estds = []
            Terrs = []
            Tstds = []
            X_ref = self.evaluate_exact(diagram)
            E_ref = torch.sum(X_ref * self.value_dict[graph.out_symbol])
            Tnorm = torch.linalg.norm(X_ref)
            for _ in range(nrun):
                X_update = torch.zeros_like(self.value_dict[graph.out_symbol])
                graph.evaluate_tensor_contraction(nsamples, X_update, jit=self.jit, dynamic=self.dynamic)
                Emean, Estd, Tstd = graph.estimate_stats(nsamples, jit=self.jit, dynamic=self.dynamic)
                Eerr = Emean - E_ref
                Terr = torch.linalg.norm(X_update - X_ref)
                Eerrs.append(Eerr)
                Estds.append(Estd)
                Terrs.append(Terr)
                Tstds.append(Tstd)
            Eerrs = torch.stack(Eerrs)
            Estds = torch.stack(Estds)
            Terrs = torch.stack(Terrs)
            Tstds = torch.stack(Tstds)
            average_Eerr = torch.std(Eerrs) * torch.sqrt(torch.tensor(nsamples))
            average_Estd = torch.sqrt(torch.mean(Estds**2))
            average_Terr = torch.sqrt(torch.mean(Terrs**2)) * torch.sqrt(torch.tensor(nsamples))
            average_Tstd = torch.sqrt(torch.mean(Tstds**2))
            Ediff = (average_Eerr - average_Estd) / average_Estd
            Tdiff = (average_Terr - average_Tstd) / average_Tstd
            print(f'{str(graph):50s}    Eerr {average_Eerr:10.6f}    Estd {average_Estd:10.6f}    diff {Ediff:10.6f}   Terr {average_Terr/Tnorm:10.6f}    Tstd {average_Tstd/Tnorm:10.6f}    diff {Tdiff:10.6f}')
        self.clear_value_dict()
        self.clear_value_cache()

    def set_freezing_mask(self, max_Eerr):
        if max_Eerr is None:
            self.mask = None
            return
        HT = self.Hovov * self.T_local_antisym
        assert HT.is_contiguous()
        thres_all = torch.exp2(torch.arange(-128, 0))
        # use binary search to find the largest one such that abs(error) < max_Eerr
        left, right = 0, len(thres_all) - 1
        while right - left > 1:
            mid = (left + right) // 2
            err = torch.abs(la.sum_below_thres(HT.reshape(-1), thres_all[mid]))
            self.log(3, thres_all[mid].item(), err.item())
            if err < max_Eerr:
                left = mid
            else:
                right = mid
        thres = thres_all[left]
        E_below = la.sum_below_thres(HT.reshape(-1), thres)
        self.mask = la.get_over_thres(HT, thres)
        num = torch.sum(self.mask)
        ratio = num / self.mask.shape.numel()
        self.log(0, 'freezing threshold', thres.item(), 'ratio', ratio.item())
        self.log(0, 'estimated energy error caused by freezing', E_below.item())

    def apply_freezing_mask(self, X):
        if self.mask is None:
            return
        X = utils.get_reference_data(X)
        X0 = self.get_T_local()
        torch.where(self.mask, X, X0, out=X)

    def flatten_X(self, X_tuple):
        X_tuple = utils.get_reference_data(X_tuple)
        X1A, X1B, X2 = X_tuple
        if X1A is None:
            X1A = torch.zeros(self.X1_shape)
        if X1B is None:
            X1B = torch.zeros(self.X1_shape)
        vec = flatten_X(X1A, X1B, X2)
        return vec

    @utils.reference_decorate
    def unflatten_X(self, vec):
        X1A, X1B, X2 = unflatten_X(vec, self.nocc_full, self.nvir_full)
        if not self.singles:
            X1A = None
            X1B = None
        return X1A, X1B, X2

    @utils.reference_decorate
    def update_diis(self, X_tuple, err):
        t1 = time.time()
        vec = self.flatten_X(X_tuple)
        t2 = time.time()
        self.log(2, 'flatten time', t2 - t1)
        vec = self.diis.update(vec, err.reshape(-1))
        t3 = time.time()
        self.log(2, 'DIIS update time', t3 - t2)
        X1A_updated, X1B_updated, X2_update = self.unflatten_X(vec)
        t4 = time.time()
        self.log(2, 'unflatten time', t4 - t3)
        #relative_err = torch.linalg.norm(amplitudes - self.diis.vecs[-1]) / torch.linalg.norm(self.diis.vecs[-1])
        #self.log(1, 'flattened X difference', relative_err.item())
        return X1A_updated, X1B_updated, X2_update

    def evaluate_exact(self, diagram: CCSD.Diagram, out=None):
        tensors = [self.value_dict[name] for name in diagram.symbols]
        result = torch.einsum(diagram.expr, *tensors)
        if out is not None:
            la.add_tensor_inplace(out, result, diagram.coeff)
            #out += result * diagram.coeff
        else:
            return result * diagram.coeff

    def exact_CCSD_update_X(self, symbol):
        X_update = torch.zeros(self.get_X_shape(symbol))
        diagrams = self.get_X_diagrams(symbol)
        for diagram in self.screen_diagrams(diagrams):
            this_begin = time.time()
            self.evaluate_exact(diagram, out=X_update)
            this_end = time.time()
            self.log(3, f'{diagram} time {this_end - this_begin:.6f}')
        X_update = self.X_apply_denominator_local(symbol, X_update)
        return X_update

    def exact_CCSD_update(self, X1, X2):
        Eij_old = self.get_Eij((X1, X1, X2))
        self.build_intermediate_tensors(X1, X2)
        X1_update = self.exact_CCSD_update_X('X1') if self.singles else None
        X2_update = self.exact_CCSD_update_X('X2')
        self.apply_freezing_mask(X2_update)
        if self.diis is not None:
            Eij_new = self.get_Eij((X1_update, X1_update, X2_update))
            Eij_diff = Eij_new - Eij_old
            X1_update, _, X2_update = self.update_diis((X1_update, None, X2_update), Eij_diff)
        return X1_update, X2_update

    def exact_CCSD(self, niters):
        X1 = self.get_X1_guess()
        X2 = self.get_X2_guess()
        E = self.get_energy((X1, X1, X2))
        self.log(0, 'init energy', E.item())
        for it in range(1, niters+1):
            begin = time.time()
            X1, X2 = self.exact_CCSD_update(X1, X2)
            end = time.time()
            E = self.get_energy((X1, X1, X2))
            self.log(0, f'it{it} energy', E.item(), 'iteration time', end - begin)
        return X1, X2

    def prepare_einsum_graphs(self, X1, X2, evaluate=True):
        self.build_intermediate_tensors(X1, X2)
        graphs_all = []
        symbols = ['X1', 'X2'] if self.singles else ['X2']
        for symbol in symbols:
            out_symbol = self.get_out_symbol(symbol)
            diagrams = self.get_X_diagrams(symbol)
            diagrams_stochastic = self.screen_diagrams(diagrams, stochastic=True)
            einsum_graphs = self.get_einsum_graphs(diagrams_stochastic, out_symbol, evaluate=evaluate)
            graphs_all.extend(einsum_graphs)
        return graphs_all

    def stochastic_CCSD_update_empty(self, symbol, *args, **kwargs):
        assert symbol == 'X1' and (not self.singles)
        yield # table
        yield torch.zeros((0, ))
        yield # sample
        yield # exact
        yield # transform
        yield None

    def stochastic_CCSD_update_X(self, symbol, compute_stds=True, timings=None):
        assert symbol in ['X1', 'X2']
        out_symbol = self.get_out_symbol(symbol)
        X_shape = self.get_X_shape(symbol)
        diagrams = self.get_X_diagrams(symbol)
        diagrams_stochastic = self.screen_diagrams(diagrams, stochastic=True)
        diagrams_exact = self.screen_diagrams(diagrams, stochastic=False)

        begin = time.time()
        einsum_graphs = self.get_einsum_graphs(diagrams_stochastic, out_symbol, evaluate=True)
        end = time.time()

        if timings is not None:
            timings['table'] += end - begin
        yield

        begin = time.time()
        stds = self.get_X_stds(symbol) if compute_stds else None
        end = time.time()
        if timings is not None:
            timings['guess'] += end - begin
        nsamples_all, jit_all = yield stds

        X_update = torch.zeros(X_shape)
        #Evar_tot = 0.0
        begin = time.time()
        # rank by nsamples descending
        for graph, nsamples, jit in sorted(zip(einsum_graphs, nsamples_all, jit_all), key=lambda x: x[1], reverse=True):
        #for graph, nsamples in zip(einsum_graphs, nsamples_all):
            this_begin = time.time()
            graph.evaluate_tensor_contraction(nsamples, X_update, jit=jit, dynamic=self.dynamic)
            #_, Estd, _ = stats
            this_end = time.time()
            self.log(3, f'{graph} time {this_end - this_begin:.6f}', 'nsamples', nsamples)
            #Evar = Estd**2 / nsamples
            #Evar_tot += Evar
        end = time.time()
        del einsum_graphs
        if timings is not None:
            timings['sample'] += end - begin
        yield

        num_unique = la.count_over_thres(X_update, 1e-8)
        self.log(1, f'{symbol} sparsity', (num_unique / X_update.numel()).item())

        begin = time.time()
        for diagram in diagrams_exact:
            this_begin = time.time()
            self.evaluate_exact(diagram, out=X_update)
            this_end = time.time()
            self.log(3, f'{diagram} time {this_end - this_begin:.6f}')
        end = time.time()
        if timings is not None:
            timings['contraction'] += end - begin
        yield

        begin = time.time()
        X_update = utils.ReferenceWrapper(X_update)
        X_update = self.X_apply_denominator_local(symbol, X_update)
        end = time.time()
        if timings is not None:
            timings['transform'] += end - begin
        yield

        yield X_update

    def stochastic_CCSD_update(self, X_tuple, return_raw=False):
        wrap_ref = isinstance(X_tuple, utils.ReferenceWrapper)
        X_tuple = utils.pop_reference_data(X_tuple)
        X1A, X1B, X2 = X_tuple
        Eij_old = self.get_Eij(X_tuple)
        del X_tuple

        X1 = None if (X1A is None and X1B is None) else (X1A + X1B) * 0.5
        del X1A, X1B

        begin = time.time()
        self.build_intermediate_tensors(X1, X2)
        end = time.time()
        t_intermediate = end - begin
        self.log(1, 'building intermediates time', t_intermediate)

        timings = {'table': 0.0, 'guess': 0.0, 'sample': 0.0, 'contraction': 0.0, 'transform': 0.0}
        if self.singles:
            g_X1A = self.stochastic_CCSD_update_X('X1', timings=timings)
            g_X1B = self.stochastic_CCSD_update_X('X1', timings=timings, compute_stds=False)
        else:
            g_X1A = self.stochastic_CCSD_update_empty('X1', timings=timings)
            g_X1B = self.stochastic_CCSD_update_empty('X1', timings=timings)
        g_X2 = self.stochastic_CCSD_update_X('X2', timings=timings)
        utils.next_all_generators((g_X1A, g_X1B, g_X2))
        self.log(1, 'probability table time', timings['table'])

        stds_X1, _, stds_X2 = utils.next_all_generators((g_X1A, g_X1B, g_X2))
        stds = torch.cat([stds_X1, stds_X2])
        nsamples = assign_nsamples(stds, self.target_error)
        if (self.min_nsamples_tot is not None) and (nsamples.sum() < self.min_nsamples_tot * 0.999):
            scale = self.min_nsamples_tot / nsamples.sum()
            self.log(1, 'scaling up nsamples from', nsamples.sum().item(), 'to', self.min_nsamples_tot)
            nsamples = nsamples * scale
        if (self.max_nsamples_tot is not None) and (nsamples.sum() > self.max_nsamples_tot * 1.001):
            scale = self.max_nsamples_tot / nsamples.sum()
            self.log(1, 'scaling down nsamples from', nsamples.sum().item(), 'to', self.max_nsamples_tot)
            nsamples = nsamples * scale
        nsamples = nsamples.to(torch.int64).tolist()
        nsamples_X1 = nsamples[0:len(stds_X1)]
        nsamples_X2 = nsamples[len(stds_X1):]
        nsamples_tot = sum(nsamples)
        self.nsamples_history.append(nsamples_tot)
        jit_X1 = [self.jit and (_nsamples / nsamples_tot >= compile_threshold) for _nsamples in nsamples_X1]
        jit_X2 = [self.jit and (_nsamples / nsamples_tot >= compile_threshold) for _nsamples in nsamples_X2]
        self.log(1, 'estimate nsamples time', timings['guess'])
        self.log(0, 'nsamples total', nsamples_tot)

        utils.send_all_generators((g_X1A, g_X1B, g_X2), ((nsamples_X1, jit_X1), (nsamples_X1, jit_X1), (nsamples_X2, jit_X2)))
        self.log(1, 'sampling time', timings['sample'])
        self.clear_value_cache()

        utils.next_all_generators((g_X1A, g_X1B, g_X2))
        self.log(1, 'contraction time', timings['contraction'])
        self.clear_value_dict()
        del X1, X2

        utils.next_all_generators((g_X1A, g_X1B, g_X2))
        self.log(1, 'transform time', timings['transform'])

        X1A, X1B, X2 = utils.next_all_generators((g_X1A, g_X1B, g_X2))
        begin = time.time()
        self.apply_freezing_mask(X2)
        end = time.time()
        t_mask = end - begin
        if self.mask is not None:
            self.log(1, 'applying mask time', end - begin)

        X_tuple = (X1A, X1B, X2)
        if return_raw:
            vec = self.flatten_X(X_tuple)
        else:
            vec = None
        if self.diis is not None:
            begin = time.time()
            Eij_new = self.get_Eij(X_tuple)
            Eij_diff = Eij_new - Eij_old
            X_tuple = self.update_diis(X_tuple, Eij_diff)
            end = time.time()
            t_diis = end - begin
            self.log(1, 'DIIS time', t_diis)
        else:
            t_diis = 0.0

        self.log(0, 'total time', t_intermediate + sum(timings.values()) + t_mask + t_diis)

        if wrap_ref:
            X_tuple = utils.ReferenceWrapper(X_tuple)
        return X_tuple, vec

    def stochastic_CCSD(self, niters, wrap_ref=False):
        X_tuple = self.get_X_tuple_guess(wrap_ref=wrap_ref)
        E = self.get_energy(X_tuple)
        self.log(0, 'init energy', E.item())
        for it in range(1, niters+1):
            begin = time.time()
            X_tuple, _ = self.stochastic_CCSD_update(X_tuple)
            end = time.time()
            E = self.get_energy(X_tuple)
            self.log(0, f'it {it} energy', E.item(), 'iteration time', end - begin)
        return X_tuple

    def get_ccsd_t(self):
        Roo = self.value_dict['Roo']
        Rov = self.value_dict['Rov']
        Rvo = self.value_dict['Rvo']
        Rvv = self.value_dict['Rvv']
        n_full = self.nocc_full + self.nvir_full
        Rfull = torch.zeros((n_full, n_full, self.naux_full))
        o = slice(0, self.nocc_full)
        v = slice(self.nocc_full, n_full)
        Rfull[o, o] = Roo
        Rfull[o, v] = Rov
        Rfull[v, o] = Rvo
        Rfull[v, v] = Rvv
        U = torch.block_diag(self.Uocc, self.Uvir)
        R_canonical = la.transform_from_left(Rfull, [U.T, U.T, None])
        e_canonical = torch.concat((self.eocc_canonical, self.evir_canonical))
        aligned = (True, True)
        block_size_occ = self.nocc_full // self.block_size if self.pad_occ else self.nocc_full
        block_size_vir = self.nvir_full // self.block_size if self.pad_vir else self.nvir_full
        block_size = (block_size_occ, block_size_vir)
        return pt.StochasticCCSDT(R_canonical, e_canonical, self.nocc_full, block_size=block_size, aligned=aligned, verbose=self.verbose)

    def initialize_pt(self, release_cc_memory=False):
        self.pt = self.get_ccsd_t()
        if release_cc_memory:
            self.clear()
        self.pt.initialize()

    def vec2canonical(self, vec):
        X1A, X1B, X2 = self.unflatten_X(vec)
        X1A = utils.ReferenceWrapper(X1A)
        X1B = utils.ReferenceWrapper(X1B)
        X2 = utils.ReferenceWrapper(X2)
        if self.singles:
            X1A_canonical = self.X1_local_to_canonical(X1A)
            X1B_canonical = self.X1_local_to_canonical(X1B)
            X1_canonical = (X1A_canonical + X1B_canonical) / 2
        else:
            X1_canonical = torch.zeros(self.X1_shape)
        X2_canonical = self.X2_local_to_canonical(X2)
        return X1_canonical, X2_canonical

    def compute_biased_pt_energy(self, vec, target_error, **pt_kwargs):
        es = self.pt.evaluate(*self.vec2canonical(vec), target_error, **pt_kwargs)
        mean = es.mean()
        std = es.std() / torch.sqrt(torch.tensor(es.shape[0]))
        return mean, std

    def compute_pt_energy(self, vec1, vec2, target_error, ratio, **pt_kwargs):
        mean_add, std_add = self.compute_biased_pt_energy((vec1 + vec2) / 2, target_error, **pt_kwargs)
        mean_minus, std_minus = self.compute_biased_pt_energy((vec1 - vec2) / 2, target_error / ratio, **pt_kwargs)
        mean = mean_add - mean_minus
        std = torch.sqrt((std_add**2 + std_minus**2))
        return mean, std
