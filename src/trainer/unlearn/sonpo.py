from __future__ import annotations
from typing import Tuple
try:
    from .npo import NPO
except Exception:
    from npo import NPO
try:
    from ..optim.sophia import SophiaG
except Exception:
    from sophia import SophiaG

class SONPO(NPO):

    def __init__(self, rho: float=0.04, sophia_betas: Tuple[float, float]=(0.9, 0.95), sophia_eps: float=1e-05, *args, **kwargs):
        self.rho = rho
        self.sophia_betas = tuple(sophia_betas)
        self.sophia_eps = sophia_eps
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad], 'weight_decay': 0.0}]
        self.optimizer = SophiaG(optimizer_grouped_parameters, lr=self.args.learning_rate, betas=self.sophia_betas, rho=self.rho, weight_decay=self.args.weight_decay, eps=self.sophia_eps)
        return self.optimizer
