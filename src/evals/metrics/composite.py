from evals.metrics.base import unlearning_metric
from tofu_scores import hmean, memorization_score, metric_value, privacy_score, utility_score


@unlearning_metric(name='paper_mem_score')
def paper_mem_score(model, **kwargs):
    return {'agg_value': memorization_score(kwargs['pre_compute'])}


@unlearning_metric(name='paper_priv_score')
def paper_priv_score(model, **kwargs):
    retain = kwargs.get('reference_logs', {}).get('retain_model_logs', {})
    return {'agg_value': privacy_score(kwargs['pre_compute'], retain)}


@unlearning_metric(name='paper_utility_score')
def paper_utility_score(model, **kwargs):
    retain = kwargs.get('reference_logs', {}).get('retain_model_logs')
    if not retain:
        raise ValueError('paper_utility_score requires the matching retain reference log')
    return {'agg_value': utility_score(kwargs['pre_compute'], retain)}


@unlearning_metric(name='paper_agg_score')
def paper_agg_score(model, **kwargs):
    pre = kwargs['pre_compute']
    return {'agg_value': hmean(metric_value(pre, key) for key in ('mem', 'priv', 'util'))}
