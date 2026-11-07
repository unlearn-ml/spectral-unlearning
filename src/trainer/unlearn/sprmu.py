from __future__ import annotations
import inspect
from typing import List, Set
import torch
import torch.nn as nn
from trainer.unlearn.rmu import RMU
from trainer.optim.soap import SOAP

class SPRMU(RMU):

    def __init__(self, soap_betas=(0.95, 0.95), soap_eps: float=1e-08, precondition_frequency: int=10, shampoo_beta: float=-1.0, max_precond_dim: int=10000, merge_dims: bool=False, precondition_1d: bool=False, normalize_grads: bool=False, data_format: str='channels_first', correct_bias: bool=True, cast_dtype: str='bfloat16', *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.cast_dtype = str(cast_dtype)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        decay_parameters = self._get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n in decay_parameters], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n not in decay_parameters], 'weight_decay': 0.0}]
        _dtype_map = {'float32': torch.float32, 'fp32': torch.float32, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
        optimizer_kwargs = dict(lr=self.args.learning_rate, betas=self.soap_betas, eps=self.soap_eps, weight_decay=self.args.weight_decay, precondition_frequency=self.precondition_frequency, shampoo_beta=self.shampoo_beta, max_precond_dim=self.max_precond_dim, merge_dims=self.merge_dims, precondition_1d=self.precondition_1d, normalize_grads=self.normalize_grads, data_format=self.data_format, correct_bias=self.correct_bias, cast_dtype=_dtype_map.get(self.cast_dtype.lower(), torch.float32))
        valid_params = set(inspect.signature(SOAP.__init__).parameters.keys())
        filtered_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in valid_params}
        self.optimizer = SOAP(optimizer_grouped_parameters, **filtered_kwargs)
        return self.optimizer

    @staticmethod
    def _get_decay_parameter_names(model: nn.Module) -> Set[str]:
        decay: List[str] = []
        for name, module in model.named_modules():
            for child_name, param in module.named_parameters(recurse=False):
                full_name = f'{name}.{child_name}' if name else child_name
                if not param.requires_grad:
                    continue
                if child_name.endswith('bias'):
                    continue
                if isinstance(module, nn.LayerNorm):
                    continue
                decay.append(full_name)
        return set(decay)
