from typing import Any, Optional, Union
import torch
from torch import nn
from copy import deepcopy
from packaging import version
from trainer.base import FinetuneTrainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from accelerate.utils import is_deepspeed_available
if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse('1.10')
    from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
if is_deepspeed_available():
    import deepspeed

class UnlearnTrainer(FinetuneTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_dynamics = {}

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        if model is not None:
            if hasattr(model, 'config'):
                hidden_size = max(model.config.hidden_sizes) if getattr(model.config, 'hidden_sizes', None) else getattr(model.config, 'hidden_size', None)
                if hidden_size is not None and config_kwargs['zero_optimization']['stage'] == 3:
                    config_kwargs.update({'zero_optimization.reduce_bucket_size': hidden_size * hidden_size, 'zero_optimization.stage3_param_persistence_threshold': 10 * hidden_size, 'zero_optimization.stage3_prefetch_bucket_size': 0.9 * hidden_size * hidden_size})
        if config_kwargs['zero_optimization']['stage'] != 3:
            config_kwargs['zero_optimization']['stage'] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def prediction_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[list[str]]=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all((inputs.get(k) is not None for k in self.label_names))
        return_loss = inputs.get('return_loss', None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, 'config'):
                ignore_keys = getattr(self.model.config, 'keys_to_ignore_at_inference', ['past_key_values'])
            else:
                ignore_keys = []
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple((inputs.get(name) for name in self.label_names)))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs['loss']
                        logits_mb = tuple((v for k, v in raw_outputs.items() if k not in ignore_keys + ['loss']))
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]
                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple((v for k, v in raw_outputs.items() if k not in ignore_keys))
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            elif has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
                loss = loss.detach().mean()
                if isinstance(outputs, dict):
                    logits = tuple((v for k, v in outputs.items() if k not in ignore_keys + ['loss']))
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple((v for k, v in outputs.items() if k not in ignore_keys))
                else:
                    logits = outputs
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
        if prediction_loss_only:
            return (loss, None, None)
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)
