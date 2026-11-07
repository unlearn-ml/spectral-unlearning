# Modified from https://github.com/KellerJordan/Muon for unlearning.
# Upstream copyright and MIT terms are in LICENSE.
import math
import os
import torch
from torch.optim import Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-07)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def _muon_update(grad: torch.Tensor, momentum_buf: torch.Tensor, beta: float=0.95, ns_steps: int=5, nesterov: bool=True) -> torch.Tensor:
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update

class SOAPMuon(Optimizer):

    def __init__(self, params, lr: float=3e-05, betas: tuple=(0.95, 0.95), eps: float=1e-08, weight_decay: float=0.01, max_precond_dim: int=10000, normalize_grads: bool=False, correct_bias: bool=True, cast_dtype: torch.dtype=torch.float32, muon_momentum: float=0.95, muon_ns_steps: int=5):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay, 'max_precond_dim': max_precond_dim, 'normalize_grads': normalize_grads, 'correct_bias': correct_bias}
        super().__init__(params, defaults)
        self._cast_dtype = cast_dtype
        self._muon_momentum = muon_momentum
        self._muon_ns_steps = muon_ns_steps
        self._last_stats: dict = {}

    @staticmethod
    def _is_muon_param(grad: torch.Tensor, max_precond_dim: int) -> bool:
        return grad.ndim == 2 and all((d <= max_precond_dim for d in grad.shape))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        collect_stats = not os.environ.get('SKIP_OPT_STATS', '')
        sum_sq_exp_avg = 0.0
        sum_sq_exp_avg_sq = 0.0
        sum_sq_update = 0.0
        sum_sq_grad = 0.0
        n_layers = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            max_precond_dim = group['max_precond_dim']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(self._cast_dtype)
                if collect_stats:
                    gn = torch.linalg.vector_norm(grad).item()
                    sum_sq_grad += gn * gn
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                use_muon = self._is_muon_param(grad, max_precond_dim)
                if use_muon:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                    state['step'] += 1
                    update = _muon_update(grad, state['momentum_buffer'], beta=self._muon_momentum, ns_steps=self._muon_ns_steps, nesterov=True)
                    update = update.to(self._cast_dtype)
                    if group['normalize_grads']:
                        update = update / (1e-30 + torch.mean(update ** 2) ** 0.5)
                    step_size = group['lr']
                    p.add_(update.to(p.dtype), alpha=-step_size)
                    if group['weight_decay'] > 0.0:
                        p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                    if collect_stats:
                        n_mb = torch.linalg.vector_norm(state['momentum_buffer']).item()
                        n_up = torch.linalg.vector_norm(update).item()
                        sum_sq_exp_avg += n_mb * n_mb
                        sum_sq_update += n_up * n_up * step_size ** 2
                        n_layers += 1
                else:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                        state['exp_avg_sq'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                    exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                    state['step'] += 1
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group['lr']
                    if group['correct_bias']:
                        bias_correction1 = 1.0 - beta1 ** state['step']
                        bias_correction2 = 1.0 - beta2 ** state['step']
                        step_size = step_size * bias_correction2 ** 0.5 / bias_correction1
                    norm_grad = exp_avg / denom
                    if group['normalize_grads']:
                        norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)
                    p.add_(norm_grad.to(p.dtype), alpha=-step_size)
                    if group['weight_decay'] > 0.0:
                        p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                    if collect_stats:
                        n_ea = torch.linalg.vector_norm(exp_avg).item()
                        n_es = torch.linalg.vector_norm(exp_avg_sq).item()
                        n_up = torch.linalg.vector_norm(norm_grad).item()
                        sum_sq_exp_avg += n_ea * n_ea
                        sum_sq_exp_avg_sq += n_es * n_es
                        sum_sq_update += n_up * n_up * step_size ** 2
                        n_layers += 1
        self._last_stats = {'opt_grad_norm': math.sqrt(sum_sq_grad) if sum_sq_grad > 0 else 0.0, 'opt_exp_avg_norm': math.sqrt(sum_sq_exp_avg) if sum_sq_exp_avg > 0 else 0.0, 'opt_exp_avg_sq_norm': math.sqrt(sum_sq_exp_avg_sq) if sum_sq_exp_avg_sq > 0 else 0.0, 'opt_update_norm': math.sqrt(sum_sq_update) if sum_sq_update > 0 else 0.0, 'opt_n_layers': n_layers}
        return loss
