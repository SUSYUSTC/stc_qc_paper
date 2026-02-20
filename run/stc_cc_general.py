#!/usr/bin/env python
import os
os.environ['PYSCF_MAX_MEMORY'] = '100000'

# Although the run-time performance is very good, the pytorch compilation is very slow.
# Fortunately, there are a cache behavior in pytorch, but there're a few things to be careful about to use it, which will be mentioned below.
# Due to the dynamic compilation feature introduced recently in pytorch, changing molecules and basis sets does not cause recompilation (with a few exceptions)

# The number of threads is set by (1) looking for environmental variable SLURM_JOB_CPUS_PER_NODE (see __init__.py), (2) if not found, use os.cpu_count()
# In case you want to change it, set SLURM_JOB_CPUS_PER_NODE to the desired number of threads as a temporal solution.
# This is because slurm sometimes gives os.cpu_count() as the total number of cores in the machine, instead of the number of cores allocated to the job
# An undesired pytorch issue is that, code with different number of threads are compiled separately.
# Currently I set the cache dir as stc_cc/torch_cache_{N}threads by nthreads N, so changing the number of threads does not mess up cache.

def make_types(*tps):
    return lambda s: tuple([tp(item) for tp, item in zip(tps, s.split(','))])


import sys
import argparse
parser = argparse.ArgumentParser(description='STC-DF-CCSD(T)')
parser.add_argument('xyzfile', type=str)
parser.add_argument('basis', type=str)
parser.add_argument('error', type=float, help='target CCSD energy error (mH)')
parser.add_argument('niters', type=int, help='number of CCSD iterations')
parser.add_argument('-charge', type=int, default=0)
parser.add_argument('-freezing', type=float, default=None, help='freeze small amplitudes for better convergence, value is the allowed total energy error (mH)')
parser.add_argument('-diis', type=int, default=None, help='DIIS space size')
parser.add_argument('-shift', type=float, default=None, help='diagonal shift of DIIS to reduce correlation between iterations')
parser.add_argument('-init_damp', type=float, default=None, help='damping factor for first iteration')
parser.add_argument('-pt', type=make_types(int, int, float), default=None, help='(T) space, naverage, error')
parser.add_argument('-log', type=str, default=None)
parser.add_argument('--pbc', action='store_true', help='enable pbc and find lattice file by replacing .xyz with .lattice')
parser.add_argument('--minX', action='store_true', help='set weights to minimize T amplitudes error, otherwise minimize energy error')
parser.add_argument('--frozen_core', action='store_true')
parser.add_argument('--ccd', action='store_true', help='perform CCD calculation')
parser.add_argument('--linear', action='store_true', help='linearized CCD calculation, must be used together with --ccd')
parser.add_argument('--exact', action='store_true')

# a few notes about the options:
# 1. currently in the code, I averaged the last m iterations (with m=4), which is verified as a good strategy (see SI). So if you want to set CCSD target error as epsilon, you should set the input error as sqrt(m) * epsilon
# 2. freezing is typically set as error / 100, which is a very safe value, while gives some benefits. But it's not an necessary option
# 3. the option pt is somewhat hard to understand and requires some more explanantion here:
#    if -pt is not set, (T) calculation is not performed; otherwise it take 3 values with (int, int, float) types, say n, m, e
#    e is simply the target error of (T)
#    currently the (T) calculation requires two "uncorrelated" CCSD iterative solutions to reduce the bias generated from stochastic CCSD results (see SI)
#    the first CCSD solution is taken by average iteration 0, 1, ..., m-1 (counted from the end), the second CCSD solution is taken by average iteration m+n-1, m+n, ..., m+n*2-2 (also counted from the end)
#    typically n=m=2 looks more than enough for most systems, while n=m=1 is enough for some simple systems but not always enough

options = parser.parse_args()
if options.log is not None:
    file = open(options.log, 'a', buffering=1)
    sys.stdout = file
else:
    file = None
if 'SLURM_JOB_ID' in os.environ:
    print('slurm job id', os.environ['SLURM_JOB_ID'])
print(*sys.argv, sep=' ')

import numpy as np
import torch
import pyscf
from stc_cc import cc, utils_pyscf, sample, stc

# some options, shoult be good for all systems and machines
stc.quota_sampling = True
cc.use_alias_for_X2vvoo = True
cc.jit_estimation = True
#cc.test_nsamples = 1000
#cc.nrun_estimation = 16
#cc.std_tol_X1 = cc.std_tol_X2 = 0.01
sample.pack_sign = True
sample.align_alias = False
utils_pyscf.make_df_eig()

import builtins
if not hasattr(builtins, 'profile'):
    profile = lambda x: x

# pyscf.df.DEFAULT_AUXBASIS
basename = os.path.basename(options.xyzfile)
assert basename[-4:] == '.xyz'
label = basename[:-4]
dir = f"data_df/{label}_{options.basis}"
os.makedirs(dir, exist_ok=True)
xyzfile = options.xyzfile
basis_name = options.basis
target_error = options.error * 1e-3
niters = options.niters
frozen_core = options.frozen_core
charge = options.charge   # I never tested this
singles = (not options.ccd)
linear = options.linear
diis_space = options.diis
diis_shift = options.shift
do_pt = options.pt is not None
if do_pt:
    pt_space, pt_naverage, pt_error = options.pt
    pt1_its = list(range(niters - pt_naverage, niters))
    pt2_its = list(range(niters - pt_naverage * 2 - pt_space, niters - pt_naverage - pt_space))
    pt_error = pt_error * 1e-3

pbc = options.pbc
if pbc:
    latticefile = f"{options.xyzfile[0:-4]}.lattice"
    lattice = np.loadtxt(latticefile)
    if basis_name.startswith('gth-'):
        pseudo = 'gth-pade'
        assert not frozen_core
    else:
        pseudo = None
    mol = pyscf.pbc.gto.Cell(atom=xyzfile, basis=utils_pyscf.get_basis(basis_name), charge=charge, a=lattice, pseudo=pseudo)
else:
    mol = pyscf.gto.Mole(atom=xyzfile, basis=utils_pyscf.get_basis(basis_name), charge=charge)
mol.build()
nao = mol.nao
if frozen_core:
    ncore = utils_pyscf.get_ncore(mol)
    print('ncore', ncore)
else:
    ncore = 0
nocc = mol.nelectron // 2
nvir = nao - nocc
o = slice(ncore, nocc)
v = slice(nocc, None)
nocc = nocc - ncore
rhf = utils_pyscf.do_scf(mol, df=True, save_df=pbc, dir=dir)
naux = rhf.with_df.get_naoaux()
print('nocc', nocc, 'nvir', nvir, 'naux', naux)

print('RHF loaded')
print('RHF total energy (Hartree)', rhf.e_tot)
print('HOMO', rhf.mo_energy[ncore + nocc - 1])
print('LUMO', rhf.mo_energy[ncore + nocc])

C = rhf.mo_coeff
Cocc = C[:, o]
Cvir = C[:, v]
C = np.concatenate((Cocc, Cvir), axis=1)

import stc_orb
PM_guess = pbc
Cocc_guess, Cvir_guess = stc_orb.get_guess(mol, Cocc, Cvir, PM=PM_guess)

# Choose one of the three basis choice in the following

# canonical basis (for demonstration only)
#Cocc_local, Cvir_local, Uaux = Cocc, Cvir, None
#key_args = dict(canonical_denominator=True, minimal_stc=False)

# mixed local-canonical basis, do all N^5 and N^6 contractions by STC, mainly for demonstration of scaling, not practically the best
#Cocc_local, Cvir_local, Uaux = stc_orb.get_Rov_mix(rhf, Cocc_guess, Cvir_guess)
#key_args = dict(canonical_denominator=False, minimal_stc=False)

# fully local basis, do all N^6 and part of N^5 contractions by STC, leave some N^5 contractions to be performed exactly (always recommended for practical calculations)
Cocc_local, Cvir_local, Uaux = stc_orb.get_Rov(rhf, Cocc_guess, Cvir_guess)
key_args = dict(canonical_denominator=True, minimal_stc=True)

Clocal = np.concatenate((Cocc_local, Cvir_local), axis=1)

# block size of probability tables. Divide a size M table to M/b blocks, and build a M/b sized alias table to sample the block index, then do a O(b) cost search within the block
# Its value balances the memory and time cost. The choice seems a little bit tricky: the sampling cost is likely not a smooth function of block size, and heavily machine-dependent
# This annoying behavior is because I can't find a good way to implement the search inside the block, so I implemented in a hard-coding and non-dynamically-jittable way.
# I have another C++ implementation (CCD only), there's no this problem there.
# Change this value leads to a full-recompilation of everything. So once tuned, don't change it.

block_size = 20
padding = (False, True, True)

# for very small systems (nocc < 20 or nvir < 20), use follows:
#block_size = None
#padding = (False, False, False)

# weights_out_power: gamma in the SI, Sec 4.1, Eq. 35
contractor_args = dict(linear=linear, singles=singles, **key_args, weights_out_power=0.5, block_size=block_size, padding=padding, diis_space=diis_space, diis_shift=diis_shift, jit=True, dynamic=True)
print('options')
for key, value in contractor_args.items():
    print(key, value)
print()

mycc = cc.StochasticCC(rhf, Clocal, target_error, Uaux=Uaux, verbose=0, **contractor_args)
if options.minX:
    mycc.weights_out_E = False
mycc.initialize()
freezing = options.freezing * 1e-3 if (options.freezing is not None) else None
mycc.set_freezing_mask(freezing)

if options.exact:
    '''
    Perform exact CCSD calculations, just for correctness checking.
    '''
    X1_ref = mycc.get_X1_guess()
    X2_ref = mycc.get_X2_guess()
    E_ref = mycc.get_energy((X1_ref, X1_ref, X2_ref))
    print('init energy', E_ref.item())
    Es_ref = [E_ref]
    for it in range(options.niters):
        X1_ref, X2_ref = mycc.exact_CCSD_update(X1_ref, X2_ref)
        E_ref = mycc.get_energy((X1_ref, X1_ref, X2_ref))
        Es_ref.append(E_ref)
        print('CCSD energy', E_ref.item())
        print()
    sys.exit()

wrap_ref = True   # some trick to save memory, to allow deleting big input tensor in function calls without leaving a reference outside the function
X_tuple = mycc.get_X_tuple_guess(wrap_ref=wrap_ref)
mycc.initialize_diis()

vec_prev = mycc.flatten_X(X_tuple)
Eij_init = mycc.get_Eij(X_tuple)
E_init = Eij_init.sum()
print('init energy', E_init.item())
Es = [E_init]
Eij_prev = Eij_init
if do_pt:
    vec_pt1 = torch.zeros(mycc.vec_size)
    vec_pt2 = torch.zeros(mycc.vec_size)
for it in range(niters):
    in_pt1 = do_pt and (it in pt1_its)
    in_pt2 = do_pt and (it in pt2_its)
    in_pt = in_pt1 or in_pt2
    X_tuple, vec_raw = mycc.stochastic_CCSD_update(X_tuple, return_raw=in_pt)
    Eij = mycc.get_Eij(X_tuple)
    Eij_diff = Eij - Eij_prev
    E = Eij.sum()
    Es.append(E)
    print(f'it{it} energy', E.item(), 'Eij update', torch.linalg.norm(Eij_diff).item())
    vec = mycc.flatten_X(X_tuple)
    if options.init_damp and (it == 0):
        vec = vec * (1 - options.init_damp) + vec_prev * options.init_damp
        X_tuple = mycc.unflatten_X(vec, wrap_ref=wrap_ref)
    print('vec diff', (torch.linalg.norm(vec - vec_prev) / torch.linalg.norm(vec_prev)).item())
    print()
    Eij_prev = Eij
    vec_prev = vec
    if in_pt1:
        vec_pt1 += vec_raw
    if in_pt2:
        vec_pt2 += vec_raw
    del vec_raw
del vec, vec_prev
if do_pt:
    vec_pt1 /= pt_naverage
    vec_pt2 /= pt_naverage
print('factor', mycc.nsamples_history[-1] / mycc.nsamples_history[0])
print()

# average the last a few iterations
print('averaged CCSD energy', torch.stack(Es[-4:]).mean().item())
if do_pt:
    # I'm sure there're are some unexpected re-compilation bugs in the (T) code, but I haven't solved it yet.
    mycc.initialize_pt(release_cc_memory=True)
    mean, std = mycc.compute_pt_energy(vec_pt1, vec_pt2, pt_error, 4)
    print('(T) energy', mean.item(), 'std', std.item())
    del vec_pt1, vec_pt2

if file is not None:
    file.close()
