from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
EVALUATOR_REGISTRY: Dict[str, Any] = {}

def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class

def _maybe_register_lm_eval():
    try:
        from evals.lm_eval import LMEvalEvaluator
    except ImportError:
        return
    _register_evaluator(LMEvalEvaluator)

def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get('handler')
    assert evaluator_handler_name is not None, ValueError(f'{name} handler not set')
    if evaluator_handler_name == 'LMEvalEvaluator' and 'LMEvalEvaluator' not in EVALUATOR_REGISTRY:
        _maybe_register_lm_eval()
        if 'LMEvalEvaluator' not in EVALUATOR_REGISTRY:
            raise ImportError(
                "LMEvalEvaluator requested but `lm-eval` is not installed. "
                "Install it with: pip install lm-eval==0.4.4"
            )
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        raise NotImplementedError(f'{evaluator_handler_name} not implemented or not registered')
    return eval_handler(eval_cfg, **kwargs)

def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators

_register_evaluator(TOFUEvaluator)
