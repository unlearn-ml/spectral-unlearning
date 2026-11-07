# Modified KL-Shampoo adaptation: https://github.com/yorkerlin/KL-Methods
# Upstream license status is recorded in NOTICES.md.
from typing import Iterable
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from math import sqrt

class SOAPKLShampoo(Optimizer):

    def __init__(self, params: Iterable[nn.parameter.Parameter], lr: float=3e-05, betas: tuple=(0.95, 0.95), eps: float=1e-08, weight_decay: float=0.01, precondition_frequency: int=10, max_dim: int=10000, cast_dtype: torch.dtype=torch.bfloat16, precondition_1d: bool=True, init_factor: float=0.1, damping: bool=False, clamping: bool=True, max_clamp_value: int=4000) -> None:
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.precondition_frequency = precondition_frequency
        self.max_dim = max_dim
        self.cast_dtype = cast_dtype
        self.precondition_1d = precondition_1d
        self.init_factor = init_factor
        self.damping = eps if damping else 0.0
        self.clamping = clamping
        self.max_clamp_value = max_clamp_value
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._last_stats = {}

    @torch.no_grad()
    def step(self, closure=None):
        import os
        collect_stats = not os.environ.get('SKIP_OPT_STATS', '')
        sum_sq_exp_avg = 0.0
        sum_sq_P = 0.0
        sum_sq_update = 0.0
        sum_sq_grad = 0.0
        n_layers = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = torch.squeeze(p.grad.to(dtype=self.cast_dtype))
                if collect_stats:
                    n = torch.linalg.vector_norm(grad).item()
                    sum_sq_grad += n * n
                state = self.state[p]
                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)
                too_large = any((d > self.max_dim for d in grad.shape))
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['use_adam'] = too_large or (grad.ndim == 1 and (not self.precondition_1d))
                    if state['use_adam']:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    else:
                        state['P'] = []
                        state['Q'] = []
                        state['l'] = []
                        if grad.ndim == 1:
                            pp = grad.unsqueeze(1) @ grad.unsqueeze(0)
                            state['P'].append(pp)
                            state['Q'].append(self._eigenbasis(pp))
                            state['l'].append(1.0 / sqrt(self.init_factor) * torch.ones_like(pp[0, :]))
                        else:
                            for idx, dim in enumerate(grad.shape):
                                if dim <= self.max_dim:
                                    rdims = [i for i in range(grad.ndim) if i != idx]
                                    pp = torch.tensordot(grad, grad, dims=[rdims] * 2)
                                    pp.mul_((1.0 - self.beta2) / grad.numel() * dim)
                                    state['P'].append(pp)
                                    state['Q'].append(self._eigenbasis(pp))
                                    state['l'].append(1.0 / sqrt(self.init_factor) * torch.ones_like(pp[0, :]))
                                else:
                                    state['P'].append(torch.empty((0,)))
                                    state['Q'].append(torch.empty((0,)))
                                    state['l'].append(torch.empty((0,)))
                    continue
                state['step'] += 1
                cur_lr = group['lr']
                cur_wd = group.get('weight_decay', 0.0)
                if state.get('use_adam', False):
                    state['exp_avg'].lerp_(grad, 1.0 - self.beta1)
                    state['exp_avg_sq'].lerp_(grad.square(), 1.0 - self.beta2)
                    bc1 = 1.0 - self.beta1 ** state['step']
                    bc2 = 1.0 - self.beta2 ** state['step']
                    update = state['exp_avg'] / bc1 / (state['exp_avg_sq'] / bc2).sqrt().add_(self.eps)
                    p.add_(update.view(p.shape).to(p.dtype), alpha=-cur_lr)
                    if cur_wd > 0:
                        p.add_(p, alpha=-cur_lr * cur_wd)
                    if collect_stats:
                        n_ea = torch.linalg.vector_norm(state['exp_avg']).item()
                        n_up = torch.linalg.vector_norm(update).item()
                        sum_sq_exp_avg += n_ea * n_ea
                        sum_sq_update += n_up * n_up * cur_lr ** 2
                        n_layers += 1
                    continue
                state['exp_avg'].lerp_(grad, 1.0 - self.beta1)
                grad_proj = self._project(state['exp_avg'], state['Q'])
                if grad.ndim == 1:
                    eigenvalues_inv = state['l'][0] ** 2
                else:
                    eigenvalues_inv = state['l'][0].view(-1, 1) * state['l'][1].view(1, -1)
                    eigenvalues_inv.div_(1.0 + eigenvalues_inv * self.eps)
                grad_final = self._project_back(grad_proj * eigenvalues_inv, state['Q'])
                step_size = -cur_lr
                p.add_(grad_final.view(p.shape).to(p.dtype), alpha=step_size)
                if cur_wd > 0:
                    p.add_(p, alpha=-cur_lr * cur_wd)
                if collect_stats:
                    n_ea = torch.linalg.vector_norm(state['exp_avg']).item()
                    n_up = torch.linalg.vector_norm(grad_final).item()
                    sum_sq_exp_avg += n_ea * n_ea
                    sum_sq_update += n_up * n_up * cur_lr ** 2
                    for pp_mat in state['P']:
                        if pp_mat.numel() > 0:
                            n_p = torch.linalg.vector_norm(pp_mat).item()
                            sum_sq_P += n_p * n_p
                    n_layers += 1
                self._update_preconditioner(grad, state)
        import math as _math
        self._last_stats = {'opt_grad_norm': _math.sqrt(sum_sq_grad) if sum_sq_grad > 0 else 0.0, 'opt_exp_avg_norm': _math.sqrt(sum_sq_exp_avg) if sum_sq_exp_avg > 0 else 0.0, 'opt_P_norm': _math.sqrt(sum_sq_P) if sum_sq_P > 0 else 0.0, 'opt_update_norm': _math.sqrt(sum_sq_update) if sum_sq_update > 0 else 0.0, 'opt_n_layers': n_layers}

    @torch.no_grad()
    def _update_preconditioner(self, grad, state):
        traces = []
        total_trace = self.damping
        for idx in range(grad.ndim):
            if state['l'][idx].numel() == 0:
                traces.append(torch.tensor(1.0))
                continue
            cur_trace = torch.mean(state['l'][idx] ** 2)
            total_trace *= cur_trace
            traces.append(cur_trace)
        if grad.ndim == 1:
            pp = grad.unsqueeze(1) @ grad.unsqueeze(0)
            state['P'][0].lerp_(pp, 1.0 - self.beta2)
            l_diag = torch.diagonal(state['Q'][0].T @ pp @ state['Q'][0])
            state['l'][0] = self._update_eigenvalue(state['l'][0], l_diag, self.damping)
        else:
            lhalf = torch.mm(grad, state['Q'][1]) * state['l'][1].view(1, -1)
            state['P'][0].lerp_(lhalf @ lhalf.T / grad.shape[1], 1.0 - self.beta2)
            rhalf = torch.mm(grad.T, state['Q'][0]) * state['l'][0].view(1, -1)
            state['P'][1].lerp_(rhalf @ rhalf.T / grad.shape[0], 1.0 - self.beta2)
            diag_half = state['Q'][0].T @ grad @ state['Q'][1]
            ldiag = torch.mean((diag_half * state['l'][1].view(1, -1)) ** 2, 1)
            rdiag = torch.mean((diag_half * state['l'][0].view(-1, 1)) ** 2, 0)
            state['l'][0] = self._update_eigenvalue(state['l'][0], ldiag, total_trace / traces[0])
            state['l'][1] = self._update_eigenvalue(state['l'][1], rdiag, total_trace / traces[1])
        if state['step'] % self.precondition_frequency == 0:
            for idx, (pp, q) in enumerate(zip(state['P'], state['Q'])):
                if pp.numel() == 0:
                    continue
                p32 = pp.clone().float()
                q32 = q.float()
                t = q32.T @ p32 @ q32
                a, b, c = (t[-2, -2].item(), t[-2, -1].item(), t[-1, -1].item())
                d = (a - c) / 2
                sgn = 1.0 if d >= 0 else -1.0
                sigma = c - sgn * b ** 2 / (abs(d) + (d ** 2 + b ** 2) ** 0.5)
                p32.diagonal().add_(-sigma)
                state['Q'][idx] = torch.linalg.qr(p32 @ q32).Q.to(q.dtype)

    def _update_eigenvalue(self, eigenvalue, update, damping=0.0):
        D = torch.squeeze(1.0 / eigenvalue ** 2).nan_to_num_(0.0, 0.0, 0.0)
        D.lerp_(update + damping, 1.0 - self.beta2)
        sqrt_inv = (1.0 / torch.sqrt(D)).nan_to_num_(0.0, 0.0, 0.0)
        if self.clamping:
            sqrt_inv = torch.clamp(sqrt_inv, max=max(10, min(D.shape[0], self.max_clamp_value)))
        return sqrt_inv

    @staticmethod
    def _eigenbasis(mat: Tensor, eps: float=1e-30):
        m = mat.clone().float()
        m.nan_to_num_(0.0, 0.0, 0.0)
        m.diagonal().add_(eps)
        try:
            Q = torch.linalg.eigh(m).eigenvectors
        except Exception:
            Q = torch.linalg.eigh(m.to(torch.float64)).eigenvectors.float()
        return Q.to(mat.dtype).flip(dims=[1])

    @staticmethod
    def _project(grad: Tensor, Q: list):
        perm = list(range(1, grad.ndim)) + [0]
        for q in Q:
            grad = torch.tensordot(grad, q, dims=[[0], [0]]) if q.numel() > 0 else grad.permute(perm)
        return grad

    @staticmethod
    def _project_back(grad: Tensor, Q: list):
        perm = list(range(1, grad.ndim)) + [0]
        for q in Q:
            grad = torch.tensordot(grad, q, dims=[[0], [1]]) if q.numel() > 0 else grad.permute(perm)
        return grad
