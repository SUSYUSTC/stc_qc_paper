'''
Structure (from high-level to low-level):
    EinsumGraphFactory -> EinsumGraph -> Graph -> spt.Tensor -> spt.TensorHead -> spt.TensorSymmetry -> spc.PermutationGroup
All tensors in this module means sympy tensors instead of torch/numpy tensors
'''
import time
import functools
from dataclasses import dataclass
import typing as tp
import numpy as np
import torch
import sympy.tensor.tensor as spt
import sympy.combinatorics as spc
from . import sample, la, utils

# quota_sampling = False if you want truely independent samples and unbiased variance estimation
# quota_sampling = True generates very small bias but typically much better runtime performance
batch_size = 20_000_000

SymmTensor = tp.Union[spt.Tensor, spt.TensMul]
index_space = spt.TensorIndexType('I')


def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result


def get_symmetry(ndim, generators):
    if len(generators) == 0:
        return spt.TensorSymmetry.no_symmetry(ndim)
    generators = [spc.Permutation(g, size=ndim + 2) for g in generators]
    G = spc.PermutationGroup(generators)
    G.schreier_sims()
    return spt.TensorSymmetry(G.base, G.strong_gens)


def in_subspace(g, subspace):
    size = g.size
    subspace = np.array(subspace)
    vec = np.array(g(tuple(range(size))))
    vec[subspace] = subspace
    return np.all(vec == np.arange(size))


def get_symmetry_subspace(ndim, generators, subspace):
    assert isinstance(subspace, tuple)
    if len(generators) == 0:
        return spt.TensorSymmetry.no_symmetry(len(subspace))
    generators = [spc.Permutation(g, size=ndim + 2) for g in generators]
    G = spc.PermutationGroup(generators)
    subG = G.subgroup_search(lambda g: in_subspace(g, subspace))
    left_space = tuple([i for i in range(ndim) if i not in subspace])
    order = subspace + left_space
    g = spc.Permutation(order, size=ndim + 2)
    reordered_generators = [g * generator * g**-1 for generator in subG.generators]
    G = spc.PermutationGroup(reordered_generators)
    return spt.TensorSymmetry(G.base, G.strong_gens)


def get_head_tensor(symbol, symmetry: spt.TensorSymmetry, check=True):
    if check:
        assert symbol[0].isupper(), "symbol must be uppercase"
        assert symbol[0] > 'A' and symbol[0] < 'Z', "symbols A and Z are reserved for inner usage, please choose other symbols"
    return spt.TensorHead(symbol, [index_space] * symmetry.rank, symmetry)


def get_tensor_indices(indices):
    return tuple(spt.TensorIndex(idx, index_space) for idx in indices)


def get_head_tensor_probe(ndim):
    # choose Z so that Z is sorted at the end in canonical form
    symmetry = spt.TensorSymmetry.no_symmetry(ndim)
    return get_head_tensor(f'Z{ndim}', symmetry, check=False)


def get_tensor_symmetric(ndim):
    return get_head_tensor(f'A{ndim}', spt.TensorSymmetry.fully_symmetric(ndim), check=False)


def get_tensor_vector():
    # choose A so that A is sorted at the end in canonical form
    return get_head_tensor(f'A', spt.TensorSymmetry.fully_symmetric(1), check=False)


@functools.lru_cache(maxsize=None)
def contract(A, B, *args):
    if len(args) > 0:
        return contract(contract(A, B), *args)
    shared_tindices = [idx for idx in A.free_indices if idx in B.free_indices]
    B = B.substitute_indices(*((idx, -idx) for idx in shared_tindices))
    return A * B
    

def intersect(a, b):
    return ''.join([idx for idx in a if idx in b])


def substract(a, b):
    return ''.join([idx for idx in a if idx not in b])


def unordered_hash(*items):
    return hash(tuple(sorted([hash(item) for item in items])))


@functools.partial(torch.compile, dynamic=True)
def partial_sum_last(array, dims, *, square):
    if square:
        array = array ** 2
    return array.sum(dim=dims)


def partial_sum(array, axes, *, square):
    ndim = array.ndim
    summed_axes = tuple([i for i in range(ndim) if i not in axes])
    order = tuple(axes) + tuple(summed_axes)
    array = torch.permute(array, order)
    summed_dims = tuple(range(len(axes), ndim))
    #return torch.abs(array**2).sum(dim=summed_dims)
    return partial_sum_last(array, summed_dims, square=square)


def default_sampler_constructor(M1_tuple, M2_tuple):
    M1, _, M1_indices = M1_tuple
    M2, _, M2_indices = M2_tuple
    out_dim = len(set(M1_indices).intersection(set(M2_indices)))
    return sample.BatchedAliasSampler(out_dim, M1, M2)


def get_symbols(tensor: SymmTensor):
    if isinstance(tensor, spt.Tensor):
        return tuple([tensor.head.name])
    elif isinstance(tensor, spt.TensMul):
        return tuple([arg.head.name for arg in tensor.args])
    else:
        raise ValueError("unexpected tensor type")


def get_symbols_signature(tensor: SymmTensor):
    symbols = get_symbols(tensor)
    symbols = [symbol for symbol in symbols if (symbol[0] not in ('A', 'O', 'Z'))]
    symbols = sorted(symbols)
    return ' '.join(symbols)


cached_canon_bp = functools.lru_cache(maxsize=None)(spt.canon_bp)


class Diagram:
    def __init__(self, expr, symbols, coeff, path=None):
        self.expr = expr
        self.symbols = symbols
        self.coeff = coeff
        self.path = path

    def __repr__(self):
        symbols = ','.join(self.symbols)
        return f"{self.expr:25s} {symbols:20s}"

    @property
    def is_stochastic(self):
        return self.path is not None


class ValueCache:
    def __init__(self, value_dict, weights_out_power):
        self.weights_out_power = weights_out_power
        self.value_dict = value_dict
        self.cached_graph_values = {}
        self.weights_out = {}
        self.weights_out_inv = {}
        self.available_weights_out_keys = set()
        self.cached_samplers = {}

    def set_weights_out(self, key):
        symbol, axes = key
        raw_value = self.value_dict[symbol]
        weights_out = torch.sqrt(partial_sum(raw_value, axes, square=True)) ** self.weights_out_power
        assert weights_out.is_contiguous()
        self.weights_out[key] = weights_out
        self.weights_out_inv[key] = 1 / weights_out
        self.available_weights_out_keys.add(key)

    def get_weights_out(self, key):
        if key not in self.available_weights_out_keys:
            self.set_weights_out(key)
        return self.weights_out[key]

    def get_weights_out_inv(self, key):
        if key not in self.available_weights_out_keys:
            self.set_weights_out(key)
        return self.weights_out_inv[key]

    def clear_symbols(self, symbols):
        symbols = set(symbols)
        for tensor in list(self.cached_graph_values.keys()):
            if set(get_symbols(tensor)).intersection(symbols):
                del self.cached_graph_values[tensor]
        for sampler_key in list(self.cached_samplers.keys()):
            tensors = [s[0] for s in sampler_key[2]]
            if any([set(get_symbols(tensor)).intersection(symbols) for tensor in tensors]):
                del self.cached_samplers[sampler_key]


class Graph(utils.Logging):
    '''
    To track and reuse intermediate tensors during contraction
    equivalent tensors under symmetries are reused
    '''
    def __init__(self, tensor: SymmTensor, prev_graphs, indices, key=None):
        self.tensor = tensor
        self.prev_graphs = prev_graphs
        self.indices = indices
        assert tensor.free_indices == set(get_tensor_indices(indices))
        self.key = key

    @classmethod
    def from_head(cls, tensorhead: spt.TensorHead, indices):
        tensor = tensorhead(*get_tensor_indices(indices))
        return cls(tensor, None, indices, tensorhead.name)

    @classmethod
    def contract(cls, graphs, indices_out):
        return cls(contract(*[g.tensor for g in graphs]), graphs, indices_out)

    def get_canonical_tensor_and_order(self):
        '''
        A canonical form a tensor under symmetries, such that tensors differ by transpose are mapped to the same canonical form
        '''
        free_indices = get_tensor_indices(self.indices)
        set(free_indices) == self.tensor.free_indices
        nfree = len(free_indices)
        tensor_probe = get_head_tensor_probe(nfree)(*free_indices)
        contracted_tensor = contract(self.tensor, tensor_probe)
        out = cached_canon_bp(contracted_tensor)
        assert out.args[-1].head.name == tensor_probe.head.name, "unexpected result for canon_bp"
        canonical_tensor = spt.TensMul(*out.args[:-1])
        out_order = out.args[-1].indices
        return canonical_tensor, out_order

    def get_keys_recursively(self):
        if self.prev_graphs is None:
            return set([self.key])
        keys = set()
        for g in self.prev_graphs:
            keys = keys.union(g.get_keys_recursively())
        return keys

    @utils.maybe_profile
    def evaluate(self, cache: ValueCache, use_cache=True, verbose=False):
        if self.prev_graphs is None:
            assert self.key is not None
            if isinstance(self.key, tuple):
                return cache.get_weights_out(self.key)
            elif isinstance(self.key, str):
                return cache.value_dict[self.key]
            else:
                raise ValueError("unexpected key type")
        canonical_tensor, order = self.get_canonical_tensor_and_order()
        prev_indices_all = [g.indices for g in self.prev_graphs]
        einsum_expr = ','.join(prev_indices_all) + '->' + self.indices
        if use_cache and (canonical_tensor in cache.cached_graph_values):
            cached_order, cached_value = cache.cached_graph_values[canonical_tensor]
            order = tuple([cached_order.index(idx) for idx in order])
            if order != tuple(range(len(order))):
                pass
            result = torch.permute(cached_value, order)
            return result
        else:
            values = [g.evaluate(cache, use_cache=use_cache, verbose=verbose) for g in self.prev_graphs]
            t1 = time.time()
            value = la.fast_einsum(einsum_expr, *values, abs=True)
            t2 = time.time()
            if verbose:
                indent = '  ' * 3
                name = 'evaluation'
                symbols_name = ','.join([get_symbols_signature(g.tensor) for g in self.prev_graphs])
                print(f'{indent}{name:30s} {einsum_expr:18s} {symbols_name:15s} time {t2 - t1:.6f}')
            if use_cache:
                cache.cached_graph_values[canonical_tensor] = (order, value)
            return value


# necessary data of EinsumGraph for stoschastic evaluation
@dataclass
class EinsumData:
    expr: str
    path: tp.List[str]
    effective_coeff: torch.Tensor


torch.utils._pytree.register_dataclass(EinsumData)


class EinsumGraph(utils.Logging):
    def __init__(self, expr, head_tensors, head_out, path, coeff, verbose=0):
        self.expr = expr
        self.head_tensors = head_tensors
        self.head_out = head_out
        self.path = path
        self.coeff = coeff
        self.n = len(self.head_tensors)
        self.marginal_indices = self.path[0].split('->')[0]
        self.expr_in, self.expr_out = expr.split('->')
        self.exprs_in = self.expr_in.split(',')
        self.graph_out = Graph.from_head(self.head_out, self.expr_out)
        self.out_symbol = self.head_out.name
        assert len(self.exprs_in) == self.n
        self.build_graph()
        self.verbose = verbose
        self.verbose_graph = verbose >= 3
    
    @classmethod
    def from_diagram(cls, diagram: Diagram, symm_dict: tp.Dict[str, spt.TensorSymmetry], weights_out_key, verbose=0):
        assert isinstance(diagram, Diagram)
        head_tensors = [get_head_tensor(symbol, symm_dict[symbol]) for symbol in diagram.symbols]
        head_out = get_head_tensor(weights_out_key, symm_dict[weights_out_key])
        einsum_graph = cls(diagram.expr, head_tensors, head_out, diagram.path, diagram.coeff, verbose=verbose)
        return einsum_graph
    
    def to_data(self) -> EinsumData:
        data = EinsumData(self.expr, self.path, self.coeff * self.norm)
        return data

    def __repr__(self):
        symbols = ','.join([head.name for head in self.head_tensors])
        return f"{self.expr:22s} {symbols:24s}"

    def get_canonical_weights_out_graph(self, weights_out_indices):
        indices_dummy = substract(self.expr_out, weights_out_indices)
        #ndummy = len(indices_dummy)
        #tensor_dummy = get_tensor_symmetric(ndummy)(*get_tensor_indices(indices_dummy))
        tensors_dummy = [get_tensor_vector()(index) for index in get_tensor_indices(indices_dummy)]
        tensor_dummy = spt.TensMul(*tensors_dummy)
        canonical_weights_out_graph = cached_canon_bp(contract(self.graph_out.tensor, tensor_dummy))
        t = canonical_weights_out_graph.args[-1]
        assert t.head.name == self.out_symbol, "unexpected result for canon_bp"
        axes = tuple([t.indices.index(idx) for idx in get_tensor_indices(weights_out_indices)])
        return canonical_weights_out_graph, axes

    def get_keys_recursively(self):
        keys = set()
        for g in self.marginals_graph:
            keys = keys.union(g.get_keys_recursively())
        return keys

    def build_graph(self):
        graphs_by_indices = {indices: Graph.from_head(head, indices) for head, indices in zip(self.head_tensors, self.exprs_in)}
        self.marginals_graph = []
        self.samplers_graph = []
        for item in self.path[::-1]:
            free, dummy = item.split('->')
            if intersect(dummy, self.expr_out):
                assert not substract(dummy, self.expr_out)
                canonical_weights_out_graph, axes = self.get_canonical_weights_out_graph(dummy)
                graphs_by_indices[dummy] = Graph(canonical_weights_out_graph, None, dummy, key=(self.head_out.name, axes))
            connected_indices = sorted([key for key in graphs_by_indices.keys() if intersect(key, dummy)])
            Gs = tuple([graphs_by_indices[key] for key in connected_indices])
            T_contract = Graph.contract(Gs, free)
            self.samplers_graph.append((Gs, (free, dummy)))
            for key in connected_indices:
                del graphs_by_indices[key]
            if not substract(free, self.marginal_indices):
                self.marginals_graph.append(T_contract)
            else:
                graphs_by_indices[free] = T_contract
        self.samplers_graph = self.samplers_graph[::-1]

    @utils.maybe_profile
    def evaluate_marginal(self, cache: ValueCache, use_cache=True):
        values = [marginal_graph.evaluate(cache, use_cache=use_cache, verbose=self.verbose_graph) for marginal_graph in self.marginals_graph]
        indices_all = [marginal_graph.indices for marginal_graph in self.marginals_graph]
        einsum_expr = ','.join(indices_all) + '->' + self.marginal_indices
        marginal = torch.einsum(einsum_expr, *values).contiguous()
        return marginal

    @utils.maybe_profile
    def evaluate_samplers(self, sampler_constructor, cache: ValueCache, use_cache=True):
        samplers = []
        for Gs, (free, dummy) in self.samplers_graph:
            full = free + dummy
            hash_keys = [G.get_canonical_tensor_and_order() + (tuple(full.index(s) for s in G.indices), ) for G in Gs]
            hash_keys = sorted(hash_keys, key=str)
            hash_key = (len(free), len(dummy), tuple(hash_keys))
            if use_cache and (hash_key in cache.cached_samplers):
                sampler = cache.cached_samplers[hash_key]
                samplers.append(sampler)
                #print(f'use cached sampler for {self.expr}, {free}->{dummy}')
                continue

            contains_all_free = [substract(free, G.indices) == '' for G in Gs]
            if any(contains_all_free):
                idx_free = contains_all_free.index(True)
                G_free = Gs[idx_free]
                value_free = G_free.evaluate(cache, use_cache=use_cache, verbose=self.verbose_graph)
                # just a transpose
                transpose_expr = f'{G_free.indices}->{free + dummy}'
                value_free = torch.einsum(transpose_expr, value_free)
                M1_tuple = (value_free, G_free.tensor, free + dummy)

                idx_dummy_all = [i for i in range(len(Gs)) if i != idx_free]
                Gs_dummy = [Gs[i] for i in idx_dummy_all]
                G_dummy = Graph.contract(Gs_dummy, dummy) if len(Gs_dummy) > 1 else Gs_dummy[0]
                values_dummy = [Gs[i].evaluate(cache, use_cache=use_cache, verbose=self.verbose_graph) for i in idx_dummy_all]
                dummy_expr = ','.join([Gs[i].indices for i in idx_dummy_all]) + '->' + dummy
                value_dummy = torch.einsum(dummy_expr, *values_dummy).contiguous()
                M2_tuple = (value_dummy, G_dummy.tensor, dummy)
                partial_weights_out_keys = [Gs[i].key for i in idx_dummy_all]
                is_weights_out = [(key in cache.available_weights_out_keys) for key in partial_weights_out_keys]
                if any(is_weights_out):
                    weights_out_inv = []
                    for i, key in enumerate(partial_weights_out_keys):
                        if is_weights_out[i]:
                            weights_out_inv.append(cache.get_weights_out_inv(key))
                        else:
                            sizes = [self.index_sizes[s] for s in Gs[i].indices]
                            size = prod(sizes)
                            weights_out_inv.append(torch.ones(size, dtype=torch.float64))
                    weights_out_inv = torch.einsum(dummy_expr, *weights_out_inv).contiguous()
                else:
                    weights_out_inv = None
                expr_name = f'{G_free.indices},{G_dummy.indices}' + f' ({free}->{dummy})'
            else:
                assert len(Gs) == 2
                if free[0] in Gs[0].indices:
                    G1, G2 = Gs
                else:
                    G2, G1 = Gs
                value1 = G1.evaluate(cache, use_cache=use_cache, verbose=self.verbose_graph)
                value2 = G2.evaluate(cache, use_cache=use_cache, verbose=self.verbose_graph)
                indices1 = G1.indices
                indices2 = G2.indices
                free1 = intersect(free, indices1)
                free2 = intersect(free, indices2)
                free1_standard = free[0:len(free1)]
                free2_standard = free[len(free1):]
                assert set(free1) == set(free1_standard)
                assert set(free2) == set(free2_standard)
                value1 = torch.einsum(f'{indices1}->{free1_standard + dummy}', value1)
                value2 = torch.einsum(f'{indices2}->{free2_standard + dummy}', value2)
                M1_tuple = (value1, G1.tensor, free1_standard + dummy)
                M2_tuple = (value2, G2.tensor, free2_standard + dummy)
                weights_out_inv = None
                expr_name = f'{G1.indices},{G2.indices}' + f' ({free}->{dummy})'
            if sampler_constructor is None:
                sampler_constructor = default_sampler_constructor
            begin = time.time()
            if use_cache:
                if hash_key not in cache.cached_samplers:
                    cache.cached_samplers[hash_key] = sampler_constructor(M1_tuple, M2_tuple)
                sampler = cache.cached_samplers[hash_key]
            else:
                sampler = sampler_constructor(M1_tuple, M2_tuple)
            if weights_out_inv is not None:
                sampler.set_weights(weights_out_inv.contiguous())
            end = time.time()
            symbols_name = ','.join([get_symbols_signature(t) for t in [M1_tuple[1], M2_tuple[1]]])
            sampler_name = sampler.__class__.__name__
            self.log(3, f'{sampler_name:30s} {expr_name:18s} {symbols_name:15s} time {end - begin:.6f}')
            samplers.append(sampler)
        return tuple(samplers)

    def build_index_sizes(self, cache: ValueCache):
        symbol_all = [head.name for head in self.head_tensors] + [self.head_out.name]
        indices_all = self.exprs_in + [self.expr_out]
        index_sizes = {}
        for symbol, indices in zip(symbol_all, indices_all):
            shape = cache.value_dict[symbol].shape
            for idx, size in zip(indices, shape):
                if idx in index_sizes:
                    assert index_sizes[idx] == size
                else:
                    index_sizes[idx] = size
        return index_sizes

    def evaluate_graph(self, sampler_constructor, cache: ValueCache, use_cache=True):
        self.index_sizes = self.build_index_sizes(cache)
        self.marginal = self.evaluate_marginal(cache, use_cache=use_cache)
        self.norm = self.marginal.sum()
        self.samplers = self.evaluate_samplers(sampler_constructor, cache, use_cache=use_cache)
        self.weights_out = cache.value_dict[self.out_symbol]

    @utils.maybe_profile
    def evaluate_tensor_contraction(self, nsamples, out, jit=False, dynamic=True):
        nsamples = max(nsamples, 100)
        random_state = sample.get_random_state()
        einsum_data = self.to_data()
        counts = sample.make_marginal_nsamples(nsamples, self.marginal)
        uindices_marginal_batch = sample.make_marginal_indices_by_batch(counts, batch_size)
        shape_output = tuple([self.index_sizes[letter] for letter in self.expr_out])

        if out is None:
            out = torch.zeros(shape_output)
        else:
            assert out.shape == shape_output

        per_sampling_add = sample.perform_sampling_add_compile if jit else sample.perform_sampling_add
        kwargs = dict(dynamic=dynamic) if jit else dict()
        for uindices_marginal in uindices_marginal_batch:
            per_sampling_add(einsum_data, self.samplers, uindices_marginal, torch.tensor(nsamples), random_state, out, **kwargs)

    def estimate_stats(self, nsamples, nrun=None, weights_out=None, jit=False, dynamic=True):
        random_state = sample.get_random_state()
        einsum_data = self.to_data()
        if nrun is not None:
            nsamples = nsamples * nrun
        counts = sample.make_marginal_nsamples(nsamples, self.marginal)
        uindices_marginal = sample.make_marginal_indices(counts, nsamples)
        shape_output = tuple([self.index_sizes[letter] for letter in self.expr_out])

        if weights_out is None:
            weights_out = self.weights_out
        else:
            assert weights_out.shape == shape_output

        evaluate_func = sample.estimate_sampling_stats_compile if jit else sample.estimate_sampling_stats
        kwargs = dict(dynamic=dynamic) if jit else dict()
        es, values = evaluate_func(einsum_data, self.samplers, uindices_marginal, random_state, weights_out=weights_out, **kwargs)
        if nrun is not None:
            es = es.view(-1, nrun)
            values = values.view(-1, nrun)
        e_mean = es.mean(dim=0)
        e_std = es.std(dim=0)
        T_std = torch.sqrt(torch.mean(values**2, dim=0))
        return e_mean, e_std, T_std
