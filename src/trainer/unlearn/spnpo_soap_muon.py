from __future__ import annotations
from typing import Set, List
import torch.nn as nn
from trainer.unlearn.npo import NPO
from trainer.optim.muon_unlearn import SOAPMuon

class SPNPO_soap_muon(NPO):

    def __init__(self, soap_betas=(0.95, 0.95), soap_eps: float=1e-08, max_precond_dim: int=10000, muon_momentum: float=0.95, muon_ns_steps: int=5, normalize_grads: bool=False, correct_bias: bool=True, cast_dtype: str='bfloat16', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soap_betas = tuple(soap_betas)
        self.soap_eps = float(soap_eps)
        self.max_precond_dim = int(max_precond_dim)
        self.muon_momentum = float(muon_momentum)
        self.muon_ns_steps = int(muon_ns_steps)
        self.normalize_grads = bool(normalize_grads)
        self.correct_bias = bool(correct_bias)
        self.cast_dtype = str(cast_dtype)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        import torch
        _dtype_map = {'float32': torch.float32, 'fp32': torch.float32, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
        decay_params = self._get_decay_names(self.model)
        groups = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n in decay_params], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n not in decay_params], 'weight_decay': 0.0}]
        self.optimizer = SOAPMuon(groups, lr=self.args.learning_rate, betas=self.soap_betas, eps=self.soap_eps, weight_decay=self.args.weight_decay, max_precond_dim=self.max_precond_dim, muon_momentum=self.muon_momentum, muon_ns_steps=self.muon_ns_steps, normalize_grads=self.normalize_grads, correct_bias=self.correct_bias, cast_dtype=_dtype_map.get(self.cast_dtype.lower(), torch.bfloat16))
        return self.optimizer

    @staticmethod
    def _get_decay_names(model: nn.Module) -> Set[str]:
        names: List[str] = []
        for name, module in model.named_modules():
            for child_name, param in module.named_parameters(recurse=False):
                full = f'{name}.{child_name}' if name else child_name
                if not param.requires_grad:
                    continue
                if child_name.endswith('bias') or isinstance(module, nn.LayerNorm):
                    continue
                names.append(full)
        return set(names)
