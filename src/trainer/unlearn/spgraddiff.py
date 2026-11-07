from __future__ import annotations
from typing import List, Set
import inspect
import torch.nn as nn
from trainer.unlearn.grad_diff import GradDiff
from trainer.optim.soap import SOAP

class SPGradDiff(GradDiff):

    def __init__(self, soap_betas=(0.95, 0.95), soap_eps: float=1e-08, precondition_frequency: int=10, shampoo_beta: float=-1.0, max_precond_dim: int=7000, merge_dims: bool=False, precondition_1d: bool=False, normalize_grads: bool=False, data_format: str='channels_first', correct_bias: bool=True, cast_dtype: str='bfloat16', *args, **kwargs):
        self.cast_dtype = str(cast_dtype)
        self.soap_betas = tuple(soap_betas)
        self.soap_eps = float(soap_eps)
        self.precondition_frequency = int(precondition_frequency)
        self.shampoo_beta = float(shampoo_beta)
        self.max_precond_dim = int(max_precond_dim)
        self.merge_dims = bool(merge_dims)
        self.precondition_1d = bool(precondition_1d)
        self.normalize_grads = bool(normalize_grads)
        self.data_format = str(data_format)
        self.correct_bias = bool(correct_bias)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        import re as _re
        if getattr(self, 'trainable_params_regex', None):
            for _p in self.model.parameters():
                _p.requires_grad = False
            for _n, _p in self.model.named_parameters():
                if any((_re.fullmatch(_pat, _n) for _pat in self.trainable_params_regex)):
                    _p.requires_grad = True
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n in decay_parameters], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n not in decay_parameters], 'weight_decay': 0.0}]
        import torch as _torch
        _dtype_map = {'float32': _torch.float32, 'fp32': _torch.float32, 'bfloat16': _torch.bfloat16, 'bf16': _torch.bfloat16}
        optimizer_kwargs = dict(lr=self.args.learning_rate, betas=self.soap_betas, eps=self.soap_eps, weight_decay=self.args.weight_decay, precondition_frequency=self.precondition_frequency, shampoo_beta=self.shampoo_beta, max_precond_dim=self.max_precond_dim, merge_dims=self.merge_dims, precondition_1d=self.precondition_1d, normalize_grads=self.normalize_grads, data_format=self.data_format, correct_bias=self.correct_bias, cast_dtype=_dtype_map.get(self.cast_dtype.lower(), _torch.bfloat16))
        optimizer_params = set(inspect.signature(SOAP.__init__).parameters.keys())
        filtered_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in optimizer_params}
        self.optimizer = SOAP(optimizer_grouped_parameters, **filtered_kwargs)
        return self.optimizer

    @staticmethod
    def get_decay_parameter_names(model: nn.Module) -> Set[str]:
        decay_parameters: List[str] = []
        for name, module in model.named_modules():
            for child_name, param in module.named_parameters(recurse=False):
                full_name = f'{name}.{child_name}' if name else child_name
                if not param.requires_grad:
                    continue
                if child_name.endswith('bias'):
                    continue
                if isinstance(module, nn.LayerNorm):
                    continue
                decay_parameters.append(full_name)
        return set(decay_parameters)
