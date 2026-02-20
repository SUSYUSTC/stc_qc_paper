import numpy as np
import torch
from . import la


def get_trace_power_torch(M: torch.Tensor, order: int) -> torch.Tensor:
    n = M.shape[0]
    result = torch.eye(n)
    for _ in range(order):
        result = result @ M
    return torch.trace(result)


class GeneralOrbLoc:
    def cayley(self, g: torch.Tensor) -> torch.Tensor:
        I = torch.eye(g.shape[0])
        # (I - g/2)^{-1} (I + g/2)
        return torch.linalg.solve(I - 0.5 * g, I + 0.5 * g)

    @property
    def nparams(self) -> int:
        raise NotImplementedError()

    def get_value(self, A_flatten: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_and_grad(self, A_flatten: torch.Tensor, mask=None):
        raise NotImplementedError()

    def optimize(self, solver_builder, tol, mask=None, verbose=True) -> np.ndarray:
        params = torch.nn.Parameter(torch.zeros(self.nparams))
        opt = solver_builder([params])

        history = []
        it = 0

        def closure():
            opt.zero_grad()
            val = self.get_value(params)
            val.backward()
            self.correct_grad(params, mask)  # apply mask and antisymmetrize grad to ensure unitary
            return val

        while True:
            val = closure()

            opt.step(closure=closure)  # in case of LBFGS

            history.append(val.item())
            res = np.std(history[-5:]) / np.abs(np.mean(history[-5:])) if len(history) > 5 else np.inf
            if verbose:
                print(it, val.item(), res)
            if res < tol:
                break
            it += 1

        return params.detach()


class OrbLoc(GeneralOrbLoc):
    n: int

    @property
    def nparams(self) -> int:
        return self.n * self.n

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def cost_function_numpy(self, u: np.ndarray) -> np.ndarray:
        u_torch = torch.from_numpy(u)
        val = self.cost_function(u_torch)
        return val.numpy()

    def cost_function_from_A(self, A: torch.Tensor) -> torch.Tensor:
        u = self.cayley(A)
        return self.cost_function(u)

    def get_value(self, A_flatten: torch.Tensor) -> torch.Tensor:
        A = A_flatten.view(self.n, self.n)
        return self.cost_function_from_A(A)

    def correct_grad(self, A_flatten: torch.nn.Parameter, mask=None):
        assert A_flatten.grad is not None
        with torch.no_grad():
            g = A_flatten.grad.view(self.n, self.n)
            if mask is not None:
                m = torch.from_numpy(mask)
                g *= m
            g.copy_((g - g.T) * 0.5)

    def run(self, solver_builder, tol, mask=None, verbose=True):
        A_opt_flat = self.optimize(solver_builder, tol, mask=mask, verbose=verbose)
        A_opt = A_opt_flat.reshape(self.n, self.n)
        u_opt = self.cayley(A_opt)
        result = u_opt.detach().numpy()
        if hasattr(self, "post_process"):
            result = self.post_process(result)
        return result


class MultiOrbLoc(GeneralOrbLoc):
    ns: tuple  # e.g., (nocc, nvir, naux)

    @property
    def nparams(self) -> int:
        return int(sum(n * n for n in self.ns))

    @property
    def shapes(self):
        return [(n, n) for n in self.ns]

    def cost_function(self, *us: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def cost_function_numpy(self, *us: np.ndarray) -> np.ndarray:
        us_torch = [torch.from_numpy(u) for u in us]
        val = self.cost_function(*us_torch)
        return val.numpy()

    def cost_function_from_A(self, *As: torch.Tensor) -> torch.Tensor:
        us = [self.cayley(A) for A in As]
        return self.cost_function(*us)

    def flatten(self, tensors):
        for n, t in zip(self.ns, tensors):
            assert t.shape == (n, n)
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    def unflatten(self, array: torch.Tensor):
        tensors = []
        offset = 0
        for (nr, nc) in self.shapes:
            size = nr * nc
            tensors.append(array[offset:offset + size].view(nr, nc))
            offset += size
        return tensors

    def get_value(self, A_flatten: torch.Tensor) -> torch.Tensor:
        As = self.unflatten(A_flatten)
        return self.cost_function_from_A(*As)

    def correct_grad(self, A_flatten: torch.nn.Parameter, masks=None):
        assert A_flatten.grad is not None
        if masks is None:
            masks = [None] * len(self.ns)
        with torch.no_grad():
            grad_blocks = self.unflatten(A_flatten.grad)
            out_blocks = []
            for mask, g in zip(masks, grad_blocks):
                if mask is not None:
                    g = g * torch.from_numpy(mask)
                g = 0.5 * (g - g.T)
                out_blocks.append(g)
            A_flatten.grad.copy_(self.flatten(out_blocks))

    def run(self, solver_builder, tol, masks=None, verbose=True):
        As_opt_flat = self.optimize(solver_builder, tol, mask=masks, verbose=verbose)
        As_opt = self.unflatten(As_opt_flat)
        us_opt = [self.cayley(A) for A in As_opt]
        result = tuple(u.detach().numpy() for u in us_opt)
        if hasattr(self, "post_process"):
            result = self.post_process(*result)
        return result


class Boys(OrbLoc):
    def __init__(self, mol, C_init):
        self.n = C_init.shape[1]
        r = mol.intor_symmetric('int1e_r', comp=3)
        r2 = mol.intor_symmetric('int1e_r2')
        r_init = np.stack([C_init.T @ x @ C_init for x in r])
        r2_init = C_init.T @ r2 @ C_init
        self.r_init = torch.from_numpy(r_init)
        self.r2_init = torch.from_numpy(r2_init)

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        r = torch.stack([u.T @ x @ u for x in self.r_init], dim=0)
        r2 = u.T @ self.r2_init @ u
        return torch.einsum('ii->', r2) - torch.einsum('xii,xii->', r, r)


class PM(OrbLoc):
    def __init__(self, mol, C_init, **kwargs):
        import pyscf.lo
        self.n = C_init.shape[1]
        proj = pyscf.lo.pipek.atomic_pops(mol, C_init, **kwargs)
        self.proj = torch.from_numpy(proj)  # [natoms (?) x n x n] per PySCF convention
        self.exponent = 2

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        # proj_diag[x,a] = sum_{i,j} proj[x,i,j] * u[i,a] * u[j,a]
        proj_diag = torch.einsum('xij,ia,ja->xa', self.proj, u, u)
        return -torch.sum(proj_diag ** self.exponent)


class BoysOVX(MultiOrbLoc):
    def __init__(self, mol, auxmol, Cocc, Cvir):
        import pyscf.gto
        self.mol = mol
        self.auxmol = auxmol
        self.nocc = Cocc.shape[1]
        self.nvir = Cvir.shape[1]
        self.naux = auxmol.nao
        Saux = auxmol.intor('int1e_ovlp')
        Caux = la.matrix_operation_numpy(Saux, lambda x: 1.0 / np.sqrt(x))

        self.o = slice(0, self.nocc)
        self.v = slice(self.nocc, self.nocc + self.nvir)
        self.x = slice(self.nocc + self.nvir, self.nocc + self.nvir + self.naux)

        CMO = np.concatenate([Cocc, Cvir], axis=1)
        self.ns = [self.nocc, self.nvir, self.naux]

        mol_r = mol.intor_symmetric('int1e_r', comp=3)
        mol_r2 = mol.intor_symmetric('int1e_r2')
        mol_r = np.stack([CMO.T @ x @ CMO for x in mol_r])
        mol_r2 = CMO.T @ mol_r2 @ CMO

        cross_r = pyscf.gto.mole.intor_cross('int1e_r', mol, auxmol, comp=3)
        cross_r2 = pyscf.gto.mole.intor_cross('int1e_r2', mol, auxmol)
        cross_r = np.stack([CMO.T @ x @ Caux for x in cross_r])
        cross_r2 = CMO.T @ cross_r2 @ Caux

        auxmol_r = auxmol.intor_symmetric('int1e_r', comp=3)
        auxmol_r2 = auxmol.intor_symmetric('int1e_r2')
        auxmol_r = np.stack([Caux.T @ x @ Caux for x in auxmol_r])
        auxmol_r2 = Caux.T @ auxmol_r2 @ Caux

        r = np.block([
            [mol_r,                   cross_r],
            [cross_r.swapaxes(1, 2),  auxmol_r]
        ])
        r2 = np.block([
            [mol_r2,     cross_r2],
            [cross_r2.T, auxmol_r2]
        ])

        self.r = torch.from_numpy(r)     # shape [3, nall, nall]
        self.r2 = torch.from_numpy(r2)   # shape [nall, nall]

    def post_process(self, Uocc, Uvir, Uaux):
        Caux_local = self.Caux @ Uaux
        j2c = self.auxmol.intor('int2c2e')
        j2c_sqrt = la.matrix_operation_numpy(j2c, lambda x: np.sqrt(x))
        j2c_local = Caux_local.T @ j2c @ Caux_local
        j2c_local_invsqrt = la.matrix_operation_numpy(j2c_local, lambda x: 1.0 / np.sqrt(x))
        Uaux_np = j2c_sqrt @ Caux_local @ j2c_local_invsqrt
        return Uocc, Uvir, Uaux_np

    def cost_function(self, uocc: torch.Tensor, uvir: torch.Tensor, uaux: torch.Tensor) -> torch.Tensor:
        nall = self.nocc + self.nvir + self.naux
        u = torch.zeros((nall, nall))
        u[self.o, self.o] = uocc
        u[self.v, self.v] = uvir
        u[self.x, self.x] = uaux

        r = torch.stack([u.T @ x @ u for x in self.r], dim=0)
        r2 = u.T @ self.r2 @ u
        return torch.einsum('ii->', r2) - torch.einsum('xii,xii->', r, r)


class Fock(OrbLoc):
    def __init__(self, F_ao, C_init, order):
        self.n = C_init.shape[1]
        F_init = C_init.T @ F_ao @ C_init
        self.F_init = torch.from_numpy(F_init)
        self.order = int(order)

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        F = u.T @ self.F_init @ u
        Fdiag = torch.diag(F)
        Foff = F - torch.diag(Fdiag)
        denom = torch.sqrt(torch.abs(Fdiag[:, None] * Fdiag[None, :]))
        Feff = torch.abs(Foff) / denom
        return (get_trace_power_torch(Feff, self.order) / self.n) ** (1.0 / self.order)


class AccurateFock(OrbLoc):
    def __init__(self, F_ao, C_init, nocc):
        self.n = C_init.shape[1]
        F_ao = F_ao
        F_init = C_init.T @ F_ao @ C_init
        self.F_init = torch.from_numpy(F_init)
        self.nocc = int(nocc)
        self.nvir = self.n - self.nocc
        self.order = 8

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        F = u.T @ self.F_init @ u
        e = torch.diag(F)
        eo = e[:self.nocc]
        ev = e[self.nocc:]
        eov = -eo[:, None] + ev[None, :]
        F = F - torch.diag(e)
        Focc = torch.abs(F[:self.nocc, :self.nocc])
        Fvir = torch.abs(F[self.nocc:, self.nocc:])

        x = torch.ones((self.nocc, self.nvir))
        for _ in range(self.order):
            x = Focc @ x + x @ Fvir
            x = x / (eov + 1e-300)
        return (torch.sum(x) / (self.nocc * self.nvir)) ** (1.0 / self.order)


class Mixed(OrbLoc):
    def __init__(self, objs, weights, reweights):
        self.objs = objs
        self.weights = list(weights)
        self.n = objs[0].n

        u0 = torch.eye(self.n)
        for i, reweight in enumerate(reweights):
            if reweight:
                with torch.no_grad():
                    base = objs[i].cost_function(u0).item()
                    self.weights[i] = self.weights[i] / base

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        return sum(w * obj.cost_function(u) for w, obj in zip(self.weights, self.objs))


class DFAux(OrbLoc):
    def __init__(self, R, triu=False, truncation=None):
        R = np.asarray(R)
        self.n = R.shape[-1]
        self.nmo = R.shape[0]

        if triu:
            iu = np.triu_indices(self.nmo, k=1)
            idg = (np.arange(self.nmo), np.arange(self.nmo))
            M = np.concatenate([R[idg], 2.0 * R[iu]])
        else:
            M = R.reshape((-1, self.n))

        if truncation is not None:
            weights = np.einsum('iP,iP->i', M, M, optimize=True)
            order = np.argsort(weights)[::-1]
            weights = weights[order]
            nselected = int(np.sum(np.cumsum(weights) / np.sum(weights) < 1 - truncation) + 1)
            select = order[:nselected]
            print('select', nselected, 'in', self.nmo ** 2)
            M = M[select]

        self.M = torch.from_numpy(M)

    def cost_function(self, u: torch.Tensor) -> torch.Tensor:
        Mu = self.M @ u
        return torch.sum(torch.sum(torch.abs(Mu), dim=0) ** 2)


class DFAll(MultiOrbLoc):
    def __init__(self, R):
        self.ns = R.shape
        self.R = torch.from_numpy(R)

    def cost_function(self, *us: torch.Tensor) -> torch.Tensor:
        R = self.R
        for n, u in zip(self.ns[::-1], us[::-1]):
            R = u.T @ R.reshape(-1, n).T
        R = R.reshape(self.ns)
        return torch.sum(torch.sum(torch.abs(R), dim=(0, 1)) ** 2)


class MultiMixed(MultiOrbLoc):
    def __init__(self, ns, objs, poss, weights, reweights):
        self.ns = ns
        self.objs = objs
        self.poss = poss
        for obj, pos in zip(objs, poss):
            if isinstance(obj, MultiOrbLoc):
                assert isinstance(pos, (list, tuple))
                for n, p in zip(obj.ns, pos):
                    assert isinstance(p, int)
                    assert self.ns[p] == n
            elif isinstance(obj, OrbLoc):
                assert isinstance(pos, int)
                assert self.ns[pos] == obj.n
        self.weights = list(weights)

        u0s = [torch.eye(n) for n in self.ns]
        for i, reweight in enumerate(reweights):
            if reweight:
                with torch.no_grad():
                    base = objs[i].cost_function(poss[i], *u0s).item()
                    self.weights[i] = self.weights[i] / base

    def get_sub_us(self, pos, *us):
        if isinstance(pos, int):
            return (us[pos], )
        elif isinstance(pos, (list, tuple)):
            return tuple([us[p] for p in pos])
        else:
            raise ValueError("pos must be int or list/tuple of int")

    def cost_function(self, *us: torch.Tensor) -> torch.Tensor:
        return sum(w * obj.cost_function(*self.get_sub_us(pos, *us)) for w, obj, pos in zip(self.weights, self.objs, self.poss))
