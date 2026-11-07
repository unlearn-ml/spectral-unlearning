import logging
from omegaconf import OmegaConf
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate
from evals.base import Evaluator
logger = logging.getLogger('evaluator')

class LMEvalEvaluator(Evaluator):

    def __init__(self, eval_cfg, **kwargs):
        self.name = 'LMEval'
        self.eval_cfg = eval_cfg
        self.tasks = OmegaConf.to_container(self.eval_cfg.tasks, resolve=True, throw_on_missing=True)
        self.task_manager = TaskManager()
        _cfg_args = eval_cfg.get('simple_evaluate_args', None)
        if _cfg_args is not None:
            _cfg_args = OmegaConf.to_container(_cfg_args, resolve=True) or {}
        else:
            _cfg_args = {}
        self.simple_evaluate_args = {**_cfg_args, **kwargs.get('simple_evaluate_args', {})}

    def prepare_model(self, model, **kwargs):
        model.eval()
        bs = self.simple_evaluate_args.get('batch_size', 1)
        max_bs = self.simple_evaluate_args.get('max_batch_size', 64)
        return HFLM(model, batch_size=bs, max_batch_size=max_bs)

    def summarize(self, eval_results: dict, task_name: str) -> dict:
        summary = {}

        def clean_metric_key(prefix: str, metric_name: str) -> str | None:
            if metric_name == 'alias':
                return None
            base = metric_name.split(',', 1)[0].strip()
            return f'{prefix}/{base}'
        if task_name in self.task_manager.all_groups:
            group_metrics = eval_results.get('groups', {}).get(task_name, {})
            for metric_name, value in group_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value
        else:
            task_metrics = eval_results.get('results', {}).get(task_name, {})
            for metric_name, value in task_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value
        return summary

    def get_task_name(self, task):
        if isinstance(task, str):
            return task
        elif isinstance(task, dict):
            if 'task' in task:
                return task.get('task')
        raise ValueError(f'Invalid task format: {task}')

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        kwargs = {'tokenizer': kwargs.get('tokenizer', None)}
        model = self.prepare_model(model, **kwargs)
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix='SUMMARY')
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        summary = self.load_logs_from_file(summary_file_path) if not overwrite else {}
        logger.info(f'***** Running {self.name} evaluation suite *****')
        logger.info(f'Fine-grained evaluations will be saved to: {logs_file_path}')
        logger.info(f'Aggregated evaluations will be summarised in: {summary_file_path}')
        for task in self.tasks:
            task_name = self.get_task_name(task)
            if not overwrite and task_name in logs and logs[task_name]:
                logger.info(f'Skipping {task_name}, already evaluated.')
                continue
            _ = logs.pop(task_name, None)
            results = simple_evaluate(model=model, tasks=[task], task_manager=self.task_manager, **self.simple_evaluate_args)
            logs.update({task_name: results['samples']})
            summary.update(self.summarize(results, task_name))
            try:
                requested = self.simple_evaluate_args.get('batch_size', None)
                summary[f'{task_name}/batch_size_requested'] = str(requested)
                actual_bs = getattr(model, 'batch_size', None)
                if actual_bs is not None:
                    summary[f'{task_name}/batch_size_actual'] = int(actual_bs)
            except Exception:
                pass
            self.save_logs(logs, logs_file_path)
            self.save_logs(summary, summary_file_path)
        return summary
