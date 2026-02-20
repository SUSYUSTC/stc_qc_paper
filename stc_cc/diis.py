import torch
from . import la


class DIIS:
    def __init__(self, vec_size, err_size, space, shift=None, verbose=False):
        self.space = space
        self.shift = shift
        self.vec_size = vec_size
        self.err_size = err_size
        self.vecs = [torch.zeros((self.vec_size, )) for _ in range(self.space - 1)]
        self.errs = [torch.zeros((self.err_size, )) for _ in range(self.space - 1)]
        self.B = torch.diag(torch.ones((self.space - 1, )) * torch.inf)
        self.verbose = verbose

    def get_coefficients(self, B):
        ones = torch.ones((self.space, ))
        B_extended = torch.zeros((self.space + 1, self.space + 1))
        B_extended[:-1, :-1] = B
        B_extended[-1, :-1] = ones
        B_extended[:-1, -1] = ones
        B_extended[-1, -1] = 0.0
        rhs = torch.zeros((self.space + 1))
        rhs[-1] = 1.0
        coeffs = torch.linalg.solve(B_extended, rhs)[:-1]
        return coeffs

    def update(self, vec, err):
        assert vec.shape == (self.vec_size, )
        assert err.shape == (self.err_size, )
        vecs = self.vecs + [vec]
        errs = self.errs + [err]
        B = torch.zeros((self.space, self.space))
        B[:-1, :-1] = self.B.clone()
        inner = torch.stack([torch.vdot(e, err) for e in errs])
        B[-1, :] = inner
        B[:, -1] = inner
        self.B = B[1:, 1:]
        self.vecs = vecs[1:]
        self.errs = errs[1:]
        if self.shift is not None:
            ones = torch.ones((self.space, ))
            ones[-1] = 0
            B = B + torch.diag(ones * self.shift)
        coeffs = self.get_coefficients(B)
        if self.verbose:
            indent = '  ' * 2
            print(f'{indent}DIIS coefficients', coeffs.numpy())
        new_vec = la.linear_combination(tuple(vecs), tuple(coeffs))
        return new_vec
