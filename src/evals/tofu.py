from evals.base import Evaluator

class TOFUEvaluator(Evaluator):

    def __init__(self, eval_cfg, **kwargs):
        super().__init__('TOFU', eval_cfg, **kwargs)

    def load_logs_from_file(self, file):
        logs = super().load_logs_from_file(file)
        for key in ('paper_mem_score', 'paper_priv_score', 'paper_utility_score', 'paper_agg_score'):
            logs.pop(key, None)
        return logs
