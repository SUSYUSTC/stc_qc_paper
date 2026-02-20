import time
import os
import torch
import numpy as np
from stc_cc import utils_pyscf, orbopt


def get_with_cache(func, path, *args, **kwargs):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True).item()
    else:
        data = func(*args, **kwargs)
        np.save(path, data)


def get_guess(mol, Cocc, Cvir, PM=True):
    Cocc_atomic = utils_pyscf.localize_atomic(mol, Cocc)
    Cvir_atomic = utils_pyscf.localize_atomic(mol, Cvir)
    if PM:
        builder_continuous = lambda params: torch.optim.LBFGS(params, max_iter=5, history_size=5, line_search_fn='strong_wolfe')
        Uocc_guess = orbopt.PM(mol, Cocc_atomic).run(builder_continuous, 1e-4)
        Uvir_guess = orbopt.PM(mol, Cvir_atomic).run(builder_continuous, 1e-4)
        Cocc_guess = Cocc_atomic @ Uocc_guess
        Cvir_guess = Cvir_atomic @ Uvir_guess
        #Cocc_guess = pyscf.lo.PM(mol, Cocc).set(verbose=4).kernel()
        #Cvir_guess = pyscf.lo.PM(mol, Cvir).set(verbose=4).kernel()
    else:
        Cocc_guess = Cocc_atomic
        Cvir_guess = Cvir_atomic
    return Cocc_guess, Cvir_guess


def get_Rov(rhf, Cocc_guess, Cvir_guess):
    begin = time.time()
    builder = lambda params: torch.optim.Rprop(params, lr=1e-3, etas=(0.5, 1.2))
    R_ao = utils_pyscf.get_DF_tensor(rhf.with_df, transpose=True)
    Rov = np.einsum('pqx,pi,qa->iax', R_ao, Cocc_guess, Cvir_guess, optimize=True)
    Uocc_local, Uvir_local, Uaux = orbopt.DFAll(Rov).run(builder, 5e-3)
    Cocc_local = Cocc_guess @ Uocc_local
    Cvir_local = Cvir_guess @ Uvir_local
    end = time.time()
    print('Orbital optimization time', end - begin)
    return Cocc_local, Cvir_local, Uaux


def get_Rov_mix(rhf, Cocc_guess, Cvir_guess, factor=1.0, order=6):
    begin = time.time()
    builder = lambda params: torch.optim.Rprop(params, lr=1e-3, etas=(0.8, 1.1))
    R_ao = utils_pyscf.get_DF_tensor(rhf.with_df, transpose=True)
    Rov = np.einsum('pqx,pi,qa->iax', R_ao, Cocc_guess, Cvir_guess, optimize=True)
    nocc, nvir, naux = Rov.shape
    dfall = orbopt.DFAll(Rov)
    F = rhf.get_fock()
    fockocc = orbopt.Fock(F, Cocc_guess, order)
    fockvir = orbopt.Fock(F, Cvir_guess, order)
    mixed_opt = orbopt.MultiMixed((nocc, nvir, naux), (dfall, fockocc, fockvir), ((0, 1, 2), 0, 1), (1.0, factor, factor), [True, False, False])
    Uocc_local, Uvir_local, Uaux = mixed_opt.run(builder, 1e-3)
    #Uocc_local, Uvir_local, Uaux = dfall.run(builder, 5e-3)
    print('Focc', fockocc.cost_function_numpy(np.eye(nocc)), '->', fockocc.cost_function_numpy(Uocc_local))
    print('Fvir', fockvir.cost_function_numpy(np.eye(nvir)), '->', fockvir.cost_function_numpy(Uvir_local))
    print('Rov', dfall.cost_function_numpy(np.eye(nocc), np.eye(nvir), np.eye(naux)), '->', dfall.cost_function_numpy(Uocc_local, Uvir_local, Uaux))

    Cocc_local = Cocc_guess @ Uocc_local
    Cvir_local = Cvir_guess @ Uvir_local
    end = time.time()
    print('Orbital optimization time', end - begin)
    return Cocc_local, Cvir_local, Uaux
