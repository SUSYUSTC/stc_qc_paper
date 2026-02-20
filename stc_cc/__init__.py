import os
this_dir = os.path.dirname(__file__)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    nthreads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
else:
    nthreads = os.cpu_count()
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(this_dir, f'torch_cache_{nthreads}threads')
print('set TORCHINDUCTOR_CACHE_DIR to', os.environ['TORCHINDUCTOR_CACHE_DIR'])
import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')
torch._dynamo.config.capture_scalar_outputs = True
#torch.fx.experimental._config.use_duck_shape = False
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024
from .utils import enable_profile, disable_profile, set_num_threads
set_num_threads(nthreads)
from . import utils, la, sample, stc, orbopt, diis, CCSD, cc, pt
