import os
from inspect import signature, Parameter
import contextlib
import functools
import builtins
has_profile = hasattr(builtins, 'profile')
if not has_profile:
    profile = lambda x: x

profile_enabled = False


def enable_profile():
    global profile_enabled
    profile_enabled = True


def disable_profile():
    global profile_enabled
    profile_enabled = False


def set_num_threads(nthreads):
    if nthreads is None and 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        nthreads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    import threadpoolctl
    backend = threadpoolctl.threadpool_info()[0]['internal_api']
    print('numpy backend', backend)
    if nthreads is not None:
        threadpoolctl.threadpool_limits(nthreads)
        import numba
        numba.set_num_threads(nthreads)
        import torch
        torch.set_num_threads(nthreads)
        print('setting number of threads to', nthreads)


@contextlib.contextmanager
def trace_recompiles():
    import torch._dynamo.logging as dlog
    reasons = []
    def handler(reason, **kw):
        print(f"[Recompile] {reason}")
        reasons.append(reason)
    old = dlog._recompilation_handlers.copy()
    dlog.set_recompilation_handlers(on_recompile=handler)
    try:
        yield reasons
    finally:
        dlog._recompilation_handlers = old


def get_nbatches(N, batch_size):
    return (N - 1) // batch_size + 1


def tree_map(fn, tree):
    from torch.utils import _pytree
    flat, spec = _pytree.tree_flatten(tree)
    flat = [fn(x) for x in flat]
    return _pytree.tree_unflatten(flat, spec)


def get_tensor_info(x):
    import torch
    if isinstance(x, torch.Tensor):
        result = (x.shape, x.stride(), x.requires_grad, x.dtype, x.storage_offset())
        if x._base is not None:
            result_base = get_tensor_info(x._base)
            return (result, result_base)
        else:
            return result
    else:
        return x


def get_static_info(tree):
    return tree_map(get_tensor_info, tree)


def modify_signature(func, args_to_add, args_to_delete, **kwargs):
    original_sig = signature(func)
    new_params = [value for value in original_sig.parameters.values() if value.name not in args_to_delete]
    for arg in args_to_add:
        new_params.append(Parameter(arg, Parameter.POSITIONAL_ONLY))
    for key, value in kwargs.items():
        new_params.append(Parameter(key, Parameter.POSITIONAL_OR_KEYWORD, default=value))
    order = [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY, Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD]
    params_reordered = sorted(new_params, key=lambda p: order.index(p.kind))
    new_sig = original_sig.replace(parameters=params_reordered)

    def decorator(func_input):
        func_input.__signature__ = new_sig
        return func_input

    return decorator


class ReferenceWrapper:
    def __init__(self, data):
        self.data = data

    def get(self):
        return self.data

    def pop(self):
        data = self.data
        del self.data
        return data


def reference_decorate(func):
    '''
    add a wrap_ref kwarg to func, if True then return ReferenceWrapper(result)
    '''
    @modify_signature(func, [], [], wrap_ref=False)
    def wrapper(*args, **kwargs):
        ref = kwargs.pop('wrap_ref', False)
        result = func(*args, **kwargs)
        if ref:
            return ReferenceWrapper(result)
        else:
            return result
    return wrapper


def pop_reference_data(data):
    if isinstance(data, ReferenceWrapper):
        return data.pop()
    else:
        return data


def get_reference_data(data):
    if isinstance(data, ReferenceWrapper):
        return data.get()
    else:
        return data


def next_all_generators(generators):
    results = []
    for gen in generators:
        results.append(next(gen))
    return tuple(results)


def send_all_generators(generators, values):
    results = []
    for gen, val in zip(generators, values):
        results.append(gen.send(val))
    return tuple(results)


def flatten_dict(d):
    return sorted([(k, v) for k, v in d.items()], key=lambda x: x[0])


def unflatten_dict(items):
    return {k: v for k, v in items}


def maybe_profile(func):
    func_profile = profile(func)

    @modify_signature(func, [], [])
    def newfunc(*args, **kwargs):
        if profile_enabled:
            return func_profile(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return newfunc

    
class Logging:
    verbose: int

    def log(self, level, label, *args):
        if self.verbose >= level:
            indent = '  ' * level
            print(f'{indent}{label}', *args)


def clear_cache():
	import ctypes
	ctypes.CDLL("libc.so.6").malloc_trim(0)


def timing(func):
    import time
    begin = time.time()
    func()
    end = time.time()
    return end - begin


def compile(func):
    import torch
    func_static = torch.compile(func, dynamic=False)
    func_dynamic = torch.compile(func, dynamic=True)

    @modify_signature(func, [], [], dynamic=True)
    def newfunc(*args, dynamic=True, **kwargs):
        if dynamic:
            return func_dynamic(*args, **kwargs)
        else:
            return func_static(*args, **kwargs)

    return newfunc


def get_memory():
    import psutil
    process = psutil.Process()
    mem = process.memory_info().rss
    return mem / 1024**2


def get_average_until_convergence(func, tol, init_nsamples):
    import torch
    nsamples = init_nsamples
    while True:
        xs = func(nsamples)
        ratios = torch.std(xs, dim=-1) / torch.mean(xs, dim=-1) / torch.sqrt(torch.tensor(xs.shape[-1]))
        assert not torch.isnan(ratios).any()
        if torch.max(ratios) < torch.tensor(tol):
            break
        nsamples *= 2
    return torch.mean(xs, dim=-1), nsamples, ratios.max().item()
