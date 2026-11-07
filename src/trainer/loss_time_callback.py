import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from data.utils import IGNORE_INDEX

def _eval_loss(model, dataset, collator, batch_size=16, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    total_loss, total_tokens = (0.0, 0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = batch['labels'][..., 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            n_tokens = (labels != IGNORE_INDEX).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)

class LossTimeCallback(TrainerCallback):

    def __init__(self):
        self.start_time = None
        self.records = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.records = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or self.start_time is None:
            return
        loss = logs.get('loss')
        if loss is None:
            return
        now = time.time()
        elapsed = round(now - self.start_time, 3)
        prev_elapsed = self.records[-1]['elapsed_sec'] if self.records else 0
        wall_clock_step = round(elapsed - prev_elapsed, 3)
        record = {'step': state.global_step, 'epoch': round(state.epoch, 4) if state.epoch else None, 'loss': loss, 'learning_rate': logs.get('learning_rate'), 'grad_norm': logs.get('grad_norm'), 'elapsed_sec': elapsed, 'wall_clock_per_step': wall_clock_step}
        trainer = kwargs.get('trainer', None)
        if trainer is None:
            for v in kwargs.values():
                if hasattr(v, '_step_dynamics'):
                    trainer = v
                    break
        if trainer is not None:
            dynamics = getattr(trainer, '_step_dynamics', None)
            if dynamics:
                record['forget_loss'] = dynamics.get('forget_loss')
                record['retain_loss'] = dynamics.get('retain_loss')
        self.records.append(record)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.start_time is None or model is None:
            return
        trainer = None
        for v in kwargs.values():
            if hasattr(v, 'train_dataset') and hasattr(v, 'data_collator'):
                trainer = v
                break
        if trainer is None:
            return
        train_ds = trainer.train_dataset
        collator = trainer.data_collator
        forget_ds = getattr(train_ds, 'forget', None)
        retain_ds = getattr(train_ds, 'retain', None)
        record = {'step': state.global_step, 'epoch': round(state.epoch, 4) if state.epoch else None, 'elapsed_sec': round(time.time() - self.start_time, 3)}
        if forget_ds and collator:
            record['forget_loss'] = round(_eval_loss(model, forget_ds, collator, batch_size=args.per_device_eval_batch_size), 6)
        if retain_ds and collator:
            record['retain_loss'] = round(_eval_loss(model, retain_ds, collator, batch_size=args.per_device_eval_batch_size), 6)
        if self.records and self.records[-1]['step'] == state.global_step:
            self.records[-1].update(record)
        else:
            self.records.append(record)

    def save(self, output_dir):
        pass
