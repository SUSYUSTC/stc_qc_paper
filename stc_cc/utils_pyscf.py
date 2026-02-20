import os
import functools
import numpy as np
import pyscf
import pyscf.lib
import pyscf.df
import pyscf.pbc.df

df_eig = False


def wrap(func, **fixed_kwargs):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        kwargs.update(fixed_kwargs)
        return func(*args, **kwargs)
    return wrapped


def matrix_operation(A, func):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = func(eigvals)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def make_df_eig():
    global df_eig
    if df_eig:
        return

    def _eig_decompose(dev, j2c, lindep=pyscf.df.incore.LINEAR_DEP_THR):
        return matrix_operation(j2c, lambda x: 1 / np.sqrt(x))

    def eigenvalue_decomposed_metric(self, j2c):
        result = matrix_operation(j2c, lambda x: 1 / np.sqrt(x))
        return result, None, 'ED'

    pyscf.df.incore.cholesky_eri = wrap(pyscf.df.incore.cholesky_eri, decompose_j2c='eig')
    pyscf.df.incore._eig_decompose = _eig_decompose
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.eigenvalue_decomposed_metric = eigenvalue_decomposed_metric
    df_eig = True


def get_basis(basis):
    '''
    cc-pvtz -> {'default': cc-pvtz}
    cc-pvtz,H:6-31g -> {'default': 'cc-pvtz', 'H': '6-31g'}
    '''
    basis_list = basis.split(',')
    basis_all = {'default': basis_list[0]}
    for string in basis_list[1:]:
        key, value = string.split(':')
        basis_all[key] = value
    return basis_all


def do_scf(mol_or_cell, df=True, save_df=False, dir=None):
    is_pbc = (not isinstance(mol_or_cell, pyscf.gto.Mole))
    if is_pbc:
        rhf = pyscf.pbc.scf.RHF(mol_or_cell)
    else:
        rhf = pyscf.scf.RHF(mol_or_cell)
    if df:
        rhf = rhf.density_fit()
    if save_df:
        assert dir is not None
    cderi_path = f"{dir}/cderi.h5" if save_df else None
    rhf.with_df = get_DF(mol_or_cell, save_path=cderi_path)
    if dir is not None:
        chkfile = f"{dir}/HF.chk"
        if os.path.exists(chkfile):
            _, attrs = pyscf.scf.chkfile.load_scf(chkfile)
            for key, value in attrs.items():
                setattr(rhf, key, value)
        else:
            rhf.chkfile = chkfile
            rhf.scf()
    else:
        rhf.scf()
    return rhf


def get_DF(mol_or_cell, save_path=None):
    is_pbc = (not isinstance(mol_or_cell, pyscf.gto.Mole))
    if is_pbc:
        with_df = pyscf.pbc.df.GDF(mol_or_cell)
    else:
        with_df = pyscf.df.DF(mol_or_cell)
    if save_path is not None:
        if os.path.exists(save_path):
            with_df._cderi = save_path
        else:
            with_df._cderi_to_save = save_path
            with_df.build()
    else:
        with_df.build()
    return with_df


def get_DF_tensor(with_df, transpose=False):
    is_pbc = hasattr(with_df, 'cell')
    if is_pbc:
        generators = (item for item, _, _ in with_df.sr_loop())
    else:
        generators = (item for item in with_df.loop())
    tensor = np.concatenate([pyscf.lib.unpack_tril(item) for item in generators])
    if transpose:
        tensor = np.moveaxis(tensor, 0, -1).copy()
    return tensor


def get_ncore(mol):
    atoms = mol.atom_charges()
    ncores = 0
    for atom in atoms:
        if atom > 36:
            assert False, "cannot deal with large molecule"
        elif atom > 18:
            ncores += 9
        elif atom > 10:
            ncores += 5
        elif atom > 2:
            ncores += 1
    return ncores


def localize_atomic(mol, C):
    import pyscf.lo
    U = pyscf.lo.boys.atomic_init_guess(mol, C)
    return C @ U


def sort_by_ao(C):
    argmax = np.argmax(np.abs(C), axis=0)
    return C[:, np.argsort(argmax)]
