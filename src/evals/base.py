import os
import json
import logging
from evals.metrics import get_metrics
logger = logging.getLogger('evaluator')

class Evaluator:

    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.metrics
        self.metrics = self.load_metrics(self.metrics_cfg)
        logger.info(f'Evaluations stored in the experiment directory: {self.eval_cfg.output_dir}')

    def get_logs_file_path(self, output_dir, suffix='EVAL'):
        logs_filename = os.path.join(output_dir, f'{self.name}_{suffix}.json')
        return logs_filename

    def load_logs_from_file(self, file):
        logs = {}
        if os.path.exists(file):
            logger.info(f'Loading existing evaluations from {file}')
            with open(file, 'r') as f:
                logs = json.load(f)
        return logs

    def save_logs(self, logs, file):
        logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, 'w') as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f'Failed to save {file}: {e}')

    def prepare_model(self, model):
        model.eval()
        return model

    def load_metrics(self, metrics_cfg):
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics:
                continue
            agg_value = metric_results.get('agg_value', None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        model = self.prepare_model(model)
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix='SUMMARY')
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        logger.info(f'***** Running {self.name} evaluation suite *****')
        logger.info(f'Fine-grained evaluations will be saved to: {logs_file_path}')
        logger.info(f'Aggregated evaluations will be summarised in: {summary_file_path}')
        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f'Skipping {metric_name}, already evaluated.')
                if 'agg_value' in logs[metric_name]:
                    logger.info(f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}")
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            logs.pop(metric_name, None)
            kwargs = {'tokenizer': kwargs.get('tokenizer', None), 'template_args': kwargs.get('template_args', None)}
            metrics_args = self.eval_cfg.metrics[metric_name]
            result = metric_fn(model, metric_name=metric_name, cache=logs, **kwargs, **metrics_args)
            if 'agg_value' in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)
        return self.summarize(logs)
