import logging
import os
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
logger = logging.getLogger(__name__)
_EVAL_PLACEHOLDER = '_EVAL_PLACEHOLDER'

class FinetuneTrainer(Trainer):

    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        if kwargs.get('eval_dataset') is None and evaluators:
            kwargs['eval_dataset'] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]]=None, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='eval', trial: Dict[str, Any]=None) -> Dict[str, float]:
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning('Custom evaluator can be run with this Trainer only when a single accelerator process is running.')
                return {}
            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            output_dir = os.path.join(run_dir, checkpoint_folder, 'evals')
            os.makedirs(output_dir, exist_ok=True)
            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {'output_dir': output_dir, 'template_args': self.template_args, 'model': self.model, 'tokenizer': self.processing_class}
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics
        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
