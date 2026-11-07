# Modified from https://github.com/nikhilvyas/SOAP for unlearning.
# Includes GaLore/PyTorch-derived helpers; see NOTICES.md and licenses/.
from __future__ import annotations
import logging
import math
import os
from itertools import chain
import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

class SOAPFast(optim.Optimizer):

    def __init__(self, params, lr: float=3e-05, betas=(0.95, 0.95), shampoo_beta: float=-1, eps: float=1e-08, weight_decay: float=0.01, precondition_frequency: int=10, max_precond_dim: int=10000, merge_dims: bool=False, precondition_1d: bool=False, normalize_grads: bool=False, data_format: str='channels_first', correct_bias: bool=True, cast_dtype: torch.dtype=torch.float32, beta2_warmup_steps: int=0, beta2_start: float=0.5, compile_kernels: bool=False):
        defaults = {'lr': lr, 'betas': betas, 'shampoo_beta': shampoo_beta, 'eps': eps, 'weight_decay': weight_decay, 'precondition_frequency': precondition_frequency, 'max_precond_dim': max_precond_dim, 'merge_dims': merge_dims, 'precondition_1d': precondition_1d, 'normalize_grads': normalize_grads, 'correct_bias': correct_bias}
        super().__init__(params, defaults)
        self._data_format = data_format
        self._cast_dtype = cast_dtype
        self._collect_stats = bool(int(os.environ.get('SOAP_STATS', '0')))
        self._last_stats: dict = {}
        self._beta2_warmup_steps = int(beta2_warmup_steps)
        self._beta2_start = float(beta2_start)
        self._global_step = 0
        self._compile_kernels = bool(compile_kernels)
        if self._compile_kernels:
            try:
                self._project = torch.compile(self._project, dynamic=True, fullgraph=False)
                self._project_back = torch.compile(self._project_back, dynamic=True, fullgraph=False)
            except Exception as e:
                logger.warning('Could not compile SOAP projection kernels (%s)', e)

    def _merge_dims(self, grad, max_precond_dim):
        if self._data_format == 'channels_last' and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr = 1
        for sh in shape:
            tmp = curr * sh
            if tmp > max_precond_dim:
                if curr > 1:
                    new_shape.append(curr)
                    curr = sh
                else:
                    new_shape.append(sh)
                    curr = 1
            else:
                curr = tmp
        if curr > 1 or len(new_shape) == 0:
            new_shape.append(curr)
        return grad.reshape(new_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if self._compile_kernels:
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except (AttributeError, RuntimeError):
                pass
        collect_stats = self._collect_stats
        sum_sq_grad = None
        sum_sq_exp_avg = None
        sum_sq_exp_avg_sq = None
        sum_sq_P = None
        sum_sq_update = None
        n_layers = 0
        self._global_step += 1
        if self._beta2_warmup_steps > 0 and self._global_step <= self._beta2_warmup_steps:
            denom_warm = max(self._beta2_warmup_steps - 1, 1)
            warmup_frac = (self._global_step - 1) / denom_warm
        else:
            warmup_frac = 1.0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2_target = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            merge_dims = group['merge_dims']
            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']
            normalize_grads = group['normalize_grads']
            correct_bias = group['correct_bias']
            shampoo_beta_target = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else beta2_target
            beta2 = self._beta2_start + (beta2_target - self._beta2_start) * warmup_frac
            shampoo_beta = self._beta2_start + (shampoo_beta_target - self._beta2_start) * warmup_frac
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(self._cast_dtype, copy=False)
                if collect_stats:
                    gn_sq = grad.pow(2).sum()
                    sum_sq_grad = gn_sq if sum_sq_grad is None else sum_sq_grad + gn_sq
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                if 'Q' not in state:
                    self._init_preconditioner(grad, state, precondition_frequency=group['precondition_frequency'], precondition_1d=precondition_1d, shampoo_beta=shampoo_beta_target, max_precond_dim=max_precond_dim, merge_dims=merge_dims)
                    self._update_preconditioner(grad, state, max_precond_dim=max_precond_dim, merge_dims=merge_dims, precondition_1d=precondition_1d, shampoo_beta=shampoo_beta)
                    continue
                grad_projected = self._project(grad, state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_projected, grad_projected, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                if correct_bias:
                    t = state['step']
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    step_size = step_size * bc2 ** 0.5 / bc1
                norm_grad = self._project_back(exp_avg / denom, state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
                if normalize_grads:
                    rms = norm_grad.pow(2).mean().sqrt().clamp_min(1e-30)
                    norm_grad = norm_grad / rms
                p.add_(norm_grad.to(p.dtype, copy=False), alpha=-step_size)
                if wd > 0.0:
                    p.add_(p, alpha=-lr * wd)
                if collect_stats:
                    ea_sq = exp_avg.pow(2).sum()
                    sum_sq_exp_avg = ea_sq if sum_sq_exp_avg is None else sum_sq_exp_avg + ea_sq
                    es_sq = exp_avg_sq.pow(2).sum()
                    sum_sq_exp_avg_sq = es_sq if sum_sq_exp_avg_sq is None else sum_sq_exp_avg_sq + es_sq
                    up_sq = norm_grad.pow(2).sum() * step_size ** 2
                    sum_sq_update = up_sq if sum_sq_update is None else sum_sq_update + up_sq
                    for gg_mat in state.get('GG', []):
                        if torch.is_tensor(gg_mat) and gg_mat.numel() > 0:
                            p_sq = gg_mat.pow(2).sum()
                            sum_sq_P = p_sq if sum_sq_P is None else sum_sq_P + p_sq
                    n_layers += 1
                self._update_preconditioner(grad, state, max_precond_dim=max_precond_dim, merge_dims=merge_dims, precondition_1d=precondition_1d, shampoo_beta=shampoo_beta)

        def _as_float(t):
            return float(t.item()) if torch.is_tensor(t) else 0.0
        self._last_stats = {'opt_grad_norm': math.sqrt(_as_float(sum_sq_grad)), 'opt_exp_avg_norm': math.sqrt(_as_float(sum_sq_exp_avg)), 'opt_exp_avg_sq_norm': math.sqrt(_as_float(sum_sq_exp_avg_sq)), 'opt_P_norm': math.sqrt(_as_float(sum_sq_P)), 'opt_update_norm': math.sqrt(_as_float(sum_sq_update)), 'opt_n_layers': n_layers}
        return loss

    def _init_preconditioner(self, grad, state, precondition_frequency=10, shampoo_beta=0.95, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
        state['GG'] = []
        grad = grad.to(self._cast_dtype, copy=False)
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=self._cast_dtype))
        else:
            if merge_dims:
                grad = self._merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=self._cast_dtype))
        state['Q'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def _project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if isinstance(mat, list):
                if len(mat) > 0:
                    grad = torch.tensordot(grad, mat, dims=[[0], [0]])
                else:
                    permute_order = list(range(1, len(grad.shape))) + [0]
                    grad = grad.permute(permute_order)
            else:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if self._data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if isinstance(mat, list):
                if len(mat) > 0:
                    grad = torch.tensordot(grad, mat, dims=[[0], [1]])
                else:
                    permute_order = list(range(1, len(grad.shape))) + [0]
                    grad = grad.permute(permute_order)
            else:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _update_preconditioner(self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False, shampoo_beta=None):
        step = state['step']
        freq = state['precondition_frequency']
        refresh_q = state['Q'] is None or (step > 0 and step % freq == 0)
        sb = state['shampoo_beta'] if shampoo_beta is None else shampoo_beta
        if refresh_q and state['Q'] is not None:
            state['exp_avg'] = self._project_back(state['exp_avg'], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - sb)
        elif merge_dims:
            new_grad = self._merge_dims(grad, max_precond_dim)
            for idx, sh in enumerate(new_grad.shape):
                if sh <= max_precond_dim:
                    outer = torch.tensordot(new_grad, new_grad, dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2)
                    state['GG'][idx].lerp_(outer, 1 - sb)
        else:
            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    outer = torch.tensordot(grad, grad, dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2)
                    state['GG'][idx].lerp_(outer, 1 - sb)
        if state['Q'] is None:
            state['Q'] = self._get_orthogonal_matrix(state['GG'])
        elif refresh_q:
            state['Q'] = self._get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
        if refresh_q and step > 0:
            state['exp_avg'] = self._project(state['exp_avg'], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)

    def _get_orthogonal_matrix(self, mat):
        final = []
        for m in mat:
            if isinstance(m, list) and len(m) == 0:
                final.append([])
                continue
            orig_device = m.device
            m_fp = m.to(torch.float32) if m.dtype != torch.float32 else m
            eye = 1e-30 * torch.eye(m_fp.shape[0], device=m_fp.device)
            try:
                _, Q = torch.linalg.eigh(m_fp + eye)
            except Exception:
                _, Q = torch.linalg.eigh(m_fp.to(torch.float64) + eye.to(torch.float64))
                Q = Q.to(torch.float32)
            Q = torch.flip(Q, [1])
            Q = Q.to(device=orig_device, dtype=self._cast_dtype)
            final.append(Q)
        return final

    def _get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        precond_list = state['GG']
        orth_list = state['Q']
        orig_shape = state['exp_avg_sq'].shape
        permuted_shape = None
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self._merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']
        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if isinstance(m, list) and len(m) == 0:
                final.append([])
                continue
            orig_device = m.device
            m_f = m.to(torch.float32) if m.dtype != torch.float32 else m
            o_f = o.to(torch.float32) if o.dtype != torch.float32 else o
            mo = m_f @ o_f
            est_eig = (o_f * mo).sum(dim=0)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_f = o_f[:, sort_idx]
            power_iter = mo[:, sort_idx]
            Q, _ = torch.linalg.qr(power_iter)
            Q = Q.to(device=orig_device, dtype=self._cast_dtype)
            final.append(Q)
        if merge_dims:
            if self._data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state['exp_avg_sq'] = exp_avg_sq
        return final
