# Modified from https://github.com/nikhilvyas/SOAP for unlearning.
# Includes GaLore/PyTorch-derived helpers; see NOTICES.md and licenses/.
import torch
import torch.optim as optim
from itertools import chain

class SOAP(optim.Optimizer):

    def __init__(self, params, lr: float=3e-05, betas=(0.95, 0.95), shampoo_beta: float=-1, eps: float=1e-08, weight_decay: float=0.01, precondition_frequency: int=10, max_precond_dim: int=10000, merge_dims: bool=False, precondition_1d: bool=False, normalize_grads: bool=False, data_format: str='channels_first', correct_bias: bool=True, cast_dtype: torch.dtype=torch.float32, soap_start_step: int=0, warmup_accumulate_gg: bool=False, grad_mix_rho: float=1.0):
        defaults = {'lr': lr, 'betas': betas, 'shampoo_beta': shampoo_beta, 'eps': eps, 'weight_decay': weight_decay, 'precondition_frequency': precondition_frequency, 'max_precond_dim': max_precond_dim, 'merge_dims': merge_dims, 'precondition_1d': precondition_1d, 'normalize_grads': normalize_grads, 'correct_bias': correct_bias}
        super().__init__(params, defaults)
        self._data_format = data_format
        self._cast_dtype = cast_dtype
        self._soap_start_step = int(soap_start_step)
        self._warmup_accumulate_gg = bool(warmup_accumulate_gg)
        self._grad_mix_rho = float(grad_mix_rho)
        self._last_stats = {}

    def merge_dims(self, grad, max_precond_dim):
        assert self._data_format in ['channels_first', 'channels_last']
        if self._data_format == 'channels_last' and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            loss = closure()
        import os
        collect_stats = not os.environ.get('SKIP_OPT_STATS', '')
        sum_sq_exp_avg = 0.0
        sum_sq_exp_avg_sq = 0.0
        sum_sq_P = 0.0
        sum_sq_update = 0.0
        sum_sq_grad = 0.0
        n_layers = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(self._cast_dtype)
                if collect_stats:
                    n = torch.linalg.vector_norm(grad).item()
                    sum_sq_grad += n * n
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                    state['exp_avg_sq'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                if self._grad_mix_rho < 1.0 and 'exp_avg_adam' not in state:
                    state['exp_avg_adam'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                    state['exp_avg_sq_adam'] = torch.zeros_like(grad, dtype=self._cast_dtype)
                if self._warmup_accumulate_gg and 'GG' not in state:
                    self.init_preconditioner(grad, state, precondition_frequency=group['precondition_frequency'], precondition_1d=group['precondition_1d'], shampoo_beta=group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group['betas'][1], max_precond_dim=group['max_precond_dim'], merge_dims=group['merge_dims'])
                if state['step'] < self._soap_start_step:
                    beta1, beta2 = group['betas']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)
                    if self._grad_mix_rho < 1.0:
                        state['exp_avg_adam'].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        state['exp_avg_sq_adam'].mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)
                    state['step'] += 1
                    if self._warmup_accumulate_gg:
                        self._accumulate_gg(grad, state, max_precond_dim=group['max_precond_dim'], merge_dims=group['merge_dims'], precondition_1d=group['precondition_1d'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group['lr']
                    if group['correct_bias']:
                        bc1 = 1.0 - beta1 ** state['step']
                        bc2 = 1.0 - beta2 ** state['step']
                        step_size = step_size * bc2 ** 0.5 / bc1
                    p.add_((exp_avg / denom).to(p.dtype), alpha=-step_size)
                    if group['weight_decay'] > 0.0:
                        p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                    if collect_stats:
                        n_ea = torch.linalg.vector_norm(exp_avg).item()
                        n_es = torch.linalg.vector_norm(exp_avg_sq).item()
                        n_up = torch.linalg.vector_norm(exp_avg / denom).item()
                        sum_sq_exp_avg += n_ea * n_ea
                        sum_sq_exp_avg_sq += n_es * n_es
                        sum_sq_update += n_up * n_up * step_size ** 2
                        n_layers += 1
                    continue
                if 'Q' not in state or state.get('Q') is None:
                    if 'GG' not in state:
                        self.init_preconditioner(grad, state, precondition_frequency=group['precondition_frequency'], precondition_1d=group['precondition_1d'], shampoo_beta=group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group['betas'][1], max_precond_dim=group['max_precond_dim'], merge_dims=group['merge_dims'])
                    self.update_preconditioner(grad, state, max_precond_dim=group['max_precond_dim'], merge_dims=group['merge_dims'], precondition_1d=group['precondition_1d'])
                    continue
                grad_projected = self.project(grad, state, merge_dims=group['merge_dims'], max_precond_dim=group['max_precond_dim'])
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                exp_avg_projected = exp_avg
                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * bias_correction2 ** 0.5 / bias_correction1
                norm_grad = self.project_back(exp_avg_projected / denom, state, merge_dims=group['merge_dims'], max_precond_dim=group['max_precond_dim'])
                if group['normalize_grads']:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)
                if self._grad_mix_rho < 1.0:
                    ea_a = state['exp_avg_adam']
                    ea_s = state['exp_avg_sq_adam']
                    ea_a.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    ea_s.mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)
                    adam_denom = ea_s.sqrt().add_(group['eps'])
                    adam_update = ea_a / adam_denom
                    norm_grad = self._grad_mix_rho * norm_grad + (1.0 - self._grad_mix_rho) * adam_update
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
                    for gg_mat in state.get('GG', []):
                        if hasattr(gg_mat, 'numel') and gg_mat.numel() > 0:
                            n_p = torch.linalg.vector_norm(gg_mat).item()
                            sum_sq_P += n_p * n_p
                    n_layers += 1
                self.update_preconditioner(grad, state, max_precond_dim=group['max_precond_dim'], merge_dims=group['merge_dims'], precondition_1d=group['precondition_1d'])
        import math as _math
        self._last_stats = {'opt_grad_norm': _math.sqrt(sum_sq_grad) if sum_sq_grad > 0 else 0.0, 'opt_exp_avg_norm': _math.sqrt(sum_sq_exp_avg) if sum_sq_exp_avg > 0 else 0.0, 'opt_exp_avg_sq_norm': _math.sqrt(sum_sq_exp_avg_sq) if sum_sq_exp_avg_sq > 0 else 0.0, 'opt_P_norm': _math.sqrt(sum_sq_P) if sum_sq_P > 0 else 0.0, 'opt_update_norm': _math.sqrt(sum_sq_update) if sum_sq_update > 0 else 0.0, 'opt_n_layers': n_layers}
        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10, shampoo_beta=0.95, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
        state['GG'] = []
        grad = grad.to(self._cast_dtype)
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=self._cast_dtype))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=self._cast_dtype))
        state['Q'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _accumulate_gg(self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state['shampoo_beta'])
            return
        if merge_dims:
            g = self.merge_dims(grad, max_precond_dim)
        else:
            g = grad
        for idx, sh in enumerate(g.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(g, g, dims=[[*chain(range(idx), range(idx + 1, len(g.shape)))]] * 2)
                state['GG'][idx].lerp_(outer_product, 1 - state['shampoo_beta'])

    def update_preconditioner(self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        if state['Q'] is not None:
            state['exp_avg'] = self.project_back(state['exp_avg'], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state['shampoo_beta'])
        elif merge_dims:
            new_grad = self.merge_dims(grad, max_precond_dim)
            for idx, sh in enumerate(new_grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(new_grad, new_grad, dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2)
                    state['GG'][idx].lerp_(outer_product, 1 - state['shampoo_beta'])
        else:
            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(grad, grad, dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2)
                    state['GG'][idx].lerp_(outer_product, 1 - state['shampoo_beta'])
        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])
        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
        if state['step'] > 0:
            state['exp_avg'] = self.project(state['exp_avg'], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)
        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except:
                _, Q = torch.linalg.eigh(m.to(torch.float32) + 1e-06 * torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        precond_list = state['GG']
        orth_list = state['Q']
        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
        orig_shape = state['exp_avg_sq'].shape
        if self._data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']
        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        if merge_dims:
            if self._data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state['exp_avg_sq'] = exp_avg_sq
        return final
