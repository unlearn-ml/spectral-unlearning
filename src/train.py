import hydra
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from trainer.loss_time_callback import LossTimeCallback
from evals import get_evaluators
from trainer.utils import seed_everything
import time
import os
import json
import torch

@hydra.main(version_base=None, config_path='../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get('mode', 'train')
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, 'Invalid model yaml passed in train config.'
    model, tokenizer = get_model(model_cfg)
    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError('Please set trainer')
    evaluators = None
    eval_cfgs = cfg.get('eval', None)
    if eval_cfgs:
        evaluators = get_evaluators(eval_cfgs=eval_cfgs, template_args=template_args, model=model, tokenizer=tokenizer)
    trainer, trainer_args = load_trainer(trainer_cfg=trainer_cfg, model=model, train_dataset=data.get('train', None), eval_dataset=data.get('eval', None), processing_class=tokenizer, data_collator=collator, evaluators=evaluators, template_args=template_args)
    if os.environ.get('TRAJ_EPOCHS', '').strip():
        from trainer.trajectory_checkpoint_callback import TrajectoryCheckpointCallback
        traj_epochs = [int(x) for x in os.environ['TRAJ_EPOCHS'].split(',') if x.strip()]
        traj_cb = TrajectoryCheckpointCallback(output_dir=trainer_args.output_dir, target_epochs=traj_epochs, trainer=trainer)
        trainer.add_callback(traj_cb)
    if trainer_args.do_train:
        start = time.time()
        trainer.train()
        train_runtime = time.time() - start
        meta = {}
        meta['method'] = trainer_cfg.get('handler', None)
        meta['task_name'] = cfg.get('task_name', None)
        meta['seed'] = getattr(trainer_args, 'seed', None)
        meta['learning_rate'] = getattr(trainer_args, 'learning_rate', None)
        meta['weight_decay'] = getattr(trainer_args, 'weight_decay', None)
        meta['num_train_epochs'] = getattr(trainer_args, 'num_train_epochs', None)
        meta['per_device_train_batch_size'] = getattr(trainer_args, 'per_device_train_batch_size', None)
        meta['gradient_accumulation_steps'] = getattr(trainer_args, 'gradient_accumulation_steps', None)
        model_args = getattr(model_cfg, 'model_args', {})
        meta['base_model_name'] = model_args.get('pretrained_model_name_or_path', None)
        meta['torch_dtype'] = model_args.get('torch_dtype', None)
        meta['attn_implementation'] = model_args.get('attn_implementation', None)
        meta['num_gpus'] = torch.cuda.device_count()
        optim = getattr(trainer_args, 'optim', None)
        meta['optimizer'] = optim.value if hasattr(optim, 'value') else optim
        lr_sched = getattr(trainer_args, 'lr_scheduler_type', None)
        meta['lr_scheduler_type'] = lr_sched.value if hasattr(lr_sched, 'value') else str(lr_sched)
        method_args_raw = trainer_cfg.get('method_args', {})
        try:
            meta['method_args'] = OmegaConf.to_container(method_args_raw, resolve=True)
        except Exception:
            meta['method_args'] = dict(method_args_raw) if method_args_raw else {}
        meta['max_precond_dim'] = meta['method_args'].get('max_precond_dim', None)
        meta['forget_split'] = cfg.get('forget_split', None)
        meta['retain_split'] = cfg.get('retain_split', None)
        trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
        total = sum((p.numel() for p in model.parameters()))
        meta['trainable_params'] = trainable
        meta['total_params'] = total
        if torch.cuda.is_available():
            meta['gpu_peak_memory_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        meta['train_runtime_sec'] = round(train_runtime, 2)
        steps = []
        for cb in trainer.callback_handler.callbacks:
            if isinstance(cb, LossTimeCallback):
                steps = cb.records
                break
        train_log = {'meta': meta, 'steps': steps}
        try:
            os.makedirs(trainer_args.output_dir, exist_ok=True)
            log_path = os.path.join(trainer_args.output_dir, 'train_log.json')
            with open(log_path, 'w') as f:
                json.dump(train_log, f, indent=2)
        except Exception:
            pass
        if os.environ.get('SKIP_MODEL_SAVE', '0') not in ('1', 'true', 'True'):
            trainer.save_state()
            trainer.save_model(trainer_args.output_dir)
        import gc as _gc
        try:
            if getattr(trainer, 'ref_model', None) is not None:
                del trainer.ref_model
                trainer.ref_model = None
        except Exception:
            pass
        try:
            if getattr(trainer, 'optimizer', None) is not None:
                try:
                    trainer.optimizer.state.clear()
                    for g in trainer.optimizer.param_groups:
                        g['params'] = []
                except Exception:
                    pass
                del trainer.optimizer
                trainer.optimizer = None
            if getattr(trainer, 'lr_scheduler', None) is not None:
                del trainer.lr_scheduler
                trainer.lr_scheduler = None
        except Exception:
            pass
        try:
            acc = getattr(trainer, 'accelerator', None)
            if acc is not None:
                if hasattr(acc, '_optimizers'):
                    acc._optimizers = []
                if hasattr(acc, '_schedulers'):
                    acc._schedulers = []
        except Exception:
            pass
        try:
            for p in trainer.model.parameters():
                p.grad = None
        except Exception:
            pass
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix='eval')
if __name__ == '__main__':
    main()
