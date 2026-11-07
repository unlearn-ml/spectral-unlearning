from __future__ import annotations
import logging
from typing import List, Set, Tuple
import torch.nn as nn
from trainer.unlearn.rmu import RMU
from trainer.optim.sophia import SophiaG
logger = logging.getLogger(__name__)

class SORMU(RMU):

    def __init__(self, rho: float=0.04, sophia_betas: Tuple[float, float]=(0.9, 0.95), sophia_eps: float=1e-05, sophia_bs: float=5120.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)
        self.sophia_betas = tuple(sophia_betas)
        self.sophia_eps = float(sophia_eps)
        self.sophia_bs = float(sophia_bs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        decay_parameters = self._get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n in decay_parameters], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and n not in decay_parameters], 'weight_decay': 0.0}]
        self.optimizer = SophiaG(optimizer_grouped_parameters, lr=self.args.learning_rate, betas=self.sophia_betas, rho=self.rho, weight_decay=self.args.weight_decay, eps=self.sophia_eps, default_bs=self.sophia_bs)
        self._freeze_all_params(self.model, True)
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
