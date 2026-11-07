# Modified from https://github.com/Liuhong99/Sophia for unlearning.
# Upstream copyright and MIT terms are in LICENSE.
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List

class SophiaG(Optimizer):

    def __init__(self, params, lr: float=0.0001, betas=(0.965, 0.99), rho: float=0.04, weight_decay: float=0.1, *, maximize: bool=False, capturable: bool=False, dynamic: bool=False, eps: float=1e-15, default_bs: float=5120.0):
        self._default_bs = float(default_bs)
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if rho < 0.0:
            raise ValueError(f'Invalid rho parameter: {rho}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if eps <= 0.0:
            raise ValueError(f'Invalid eps value: {eps}')
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, maximize=maximize, capturable=capturable, dynamic=dynamic, eps=eps)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('dynamic', False)
            group.setdefault('eps', 1e-15)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0].get('step', None))
        if not step_is_tensor:
            for s in state_values:
                if 'step' in s:
                    s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def update_exp_avg(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if bs is None:
            bs = getattr(self, '_default_bs', 1.0)
        self.update_hessian()
        self.update_exp_avg()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])
            self._sophiag(params_with_grad, grads, exp_avgs, hessian, state_steps, bs=bs, beta1=beta1, beta2=beta2, rho=group['rho'], lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], maximize=group['maximize'], capturable=group['capturable'])
        return loss

    def _sophiag(self, params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], hessian: List[Tensor], state_steps: List[Tensor], *, bs, beta1: float, beta2: float, rho: float, lr: float, weight_decay: float, eps: float, maximize: bool, capturable: bool):
        if not all((isinstance(t, torch.Tensor) for t in state_steps)):
            raise RuntimeError('state_steps must contain singleton tensors')
        self._single_tensor_sophiag(params, grads, exp_avgs, hessian, state_steps, bs=bs, beta1=beta1, beta2=beta2, rho=rho, lr=lr, weight_decay=weight_decay, eps=eps, maximize=maximize, capturable=capturable)

    def _single_tensor_sophiag(self, params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], hessian: List[Tensor], state_steps: List[Tensor], *, bs, beta1: float, beta2: float, rho: float, lr: float, weight_decay: float, eps: float, maximize: bool, capturable: bool):
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]
            if capturable and torch.is_tensor(bs):
                assert param.is_cuda and step_t.is_cuda and bs.is_cuda
            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                hess = torch.view_as_real(hess)
                param = torch.view_as_real(param)
            step_t += 1
            param.mul_(1 - lr * weight_decay)
            denom = rho * bs * hess + eps
            ratio = (exp_avg.abs() / denom).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=-lr)
