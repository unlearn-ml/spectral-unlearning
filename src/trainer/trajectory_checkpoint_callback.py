import json
import os
import time
import torch
from transformers import TrainerCallback

class TrajectoryCheckpointCallback(TrainerCallback):

    def __init__(self, output_dir, target_epochs, trainer=None):
        self.output_dir = output_dir
        self.target_epochs = sorted(set((int(e) for e in target_epochs)))
        self.trainer = trainer
        self.start_time = None
        self.epoch_walltime = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, 'trajectory_targets.json'), 'w') as f:
                json.dump({'target_epochs': self.target_epochs}, f, indent=2)
        except Exception:
            pass

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_int = int(round(state.epoch))
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        self.epoch_walltime[epoch_int] = elapsed
        if epoch_int in self.target_epochs and self.trainer is not None:
            ckpt_path = os.path.join(self.output_dir, f'ckpt_e{epoch_int}')
            try:
                os.makedirs(ckpt_path, exist_ok=True)
                self.trainer.save_model(ckpt_path)
                meta = {'epoch': epoch_int, 'wall_clock_sec': elapsed, 'wall_clock_min': elapsed / 60.0, 'global_step': state.global_step}
                with open(os.path.join(ckpt_path, 'trajectory_meta.json'), 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f'[TrajCkpt] saved ckpt_e{epoch_int} step={state.global_step} wall={elapsed:.1f}s ({elapsed / 60:.2f}min)', flush=True)
            except Exception as e:
                print(f'[TrajCkpt] FAILED ckpt_e{epoch_int}: {e}', flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        final_target = max(self.target_epochs) if self.target_epochs else None
        if final_target is not None and self.trainer is not None:
            ckpt_path = os.path.join(self.output_dir, f'ckpt_e{final_target}')
            cfg_path = os.path.join(ckpt_path, 'config.json')
            if not os.path.exists(cfg_path):
                try:
                    os.makedirs(ckpt_path, exist_ok=True)
                    self.trainer.save_model(ckpt_path)
                    elapsed = time.time() - self.start_time if self.start_time else 0.0
                    self.epoch_walltime[final_target] = elapsed
                    meta = {'epoch': final_target, 'wall_clock_sec': elapsed, 'wall_clock_min': elapsed / 60.0, 'global_step': state.global_step, 'saved_in': 'on_train_end_fallback'}
                    with open(os.path.join(ckpt_path, 'trajectory_meta.json'), 'w') as f:
                        json.dump(meta, f, indent=2)
                    print(f'[TrajCkpt] (fallback) saved ckpt_e{final_target} step={state.global_step} wall={elapsed:.1f}s', flush=True)
                except Exception as e:
                    print(f'[TrajCkpt] (fallback) FAILED ckpt_e{final_target}: {e}', flush=True)
        try:
            with open(os.path.join(self.output_dir, 'trajectory_walltime.json'), 'w') as f:
                json.dump(self.epoch_walltime, f, indent=2)
            print(f'[TrajCkpt] wrote trajectory_walltime.json with {len(self.epoch_walltime)} epochs', flush=True)
        except Exception:
            pass
