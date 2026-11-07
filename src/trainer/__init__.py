import logging
from typing import Any, Dict
import torch
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments
from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.sograddiff import SOGradDiff
from trainer.unlearn.spgraddiff import SPGradDiff
from trainer.unlearn.spgraddiff_soap_muon import SPGradDiff_soap_muon
from trainer.unlearn.spgraddiff_soap_klshampoo import SPGradDiff_soap_klshampoo
from trainer.unlearn.npo import NPO
from trainer.unlearn.sonpo import SONPO
from trainer.unlearn.spnpo import SPNPO
from trainer.unlearn.spnpo_soap_muon import SPNPO_soap_muon
from trainer.unlearn.spnpo_soap_klshampoo import SPNPO_soap_klshampoo
from trainer.unlearn.rmu import RMU
from trainer.unlearn.sormu import SORMU
from trainer.unlearn.sprmu import SPRMU
from trainer.unlearn.sprmu_soap_muon import SPRMU_soap_muon
from trainer.unlearn.sprmu_soap_klshampoo import SPRMU_soap_klshampoo
from trainer.loss_time_callback import LossTimeCallback
logger = logging.getLogger(__name__)
TRAINER_REGISTRY: Dict[str, Any] = {}

def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class

def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop('warmup_epochs', None)
    if warmup_epochs:
        batch_size = trainer_args['per_device_train_batch_size']
        grad_accum_steps = trainer_args['gradient_accumulation_steps']
        num_devices = max(torch.cuda.device_count(), 1)
        dataset_len = len(dataset)
        trainer_args['warmup_steps'] = int(warmup_epochs * dataset_len // (batch_size * grad_accum_steps * num_devices))
    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args

def load_trainer(trainer_cfg: DictConfig, model, train_dataset=None, eval_dataset=None, processing_class=None, data_collator=None, evaluators=None, template_args=None):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get('method_args', {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get('handler')
    assert trainer_handler_name is not None, ValueError(f'{trainer_handler_name} handler not set')
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(f'{trainer_handler_name} not implemented or not registered')
    trainer = trainer_cls(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=processing_class, data_collator=data_collator, args=trainer_args, evaluators=evaluators, template_args=template_args, callbacks=[LossTimeCallback()], **method_args)
    logger.info(f'{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}')
    return (trainer, trainer_args)
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)
_register_trainer(GradDiff)
_register_trainer(SOGradDiff)
_register_trainer(SPGradDiff)
_register_trainer(SPGradDiff_soap_muon)
_register_trainer(SPGradDiff_soap_klshampoo)
_register_trainer(NPO)
_register_trainer(SONPO)
_register_trainer(SPNPO)
_register_trainer(SPNPO_soap_muon)
_register_trainer(SPNPO_soap_klshampoo)
_register_trainer(RMU)
_register_trainer(SORMU)
_register_trainer(SPRMU)
_register_trainer(SPRMU_soap_muon)
_register_trainer(SPRMU_soap_klshampoo)
TRAINER_REGISTRY['sograddiff'] = SOGradDiff
TRAINER_REGISTRY['spgraddiff'] = SPGradDiff
