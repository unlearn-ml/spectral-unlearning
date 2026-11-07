from statistics import fmean, harmonic_mean


MIA_KEYS = ('mia_loss', 'mia_zlib', 'mia_min_k', 'mia_min_k_plus_plus')
UTILITY_KEYS = (
    'retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio',
    'ra_Q_A_Prob_normalised', 'ra_Q_A_ROUGE', 'ra_Truth_Ratio',
    'wf_Q_A_Prob_normalised', 'wf_Q_A_ROUGE', 'wf_Truth_Ratio',
    'forget_Q_A_gibberish',
)
EPS = 1e-12


def hmean(values):
    return harmonic_mean(max(EPS, value) for value in values)


def clip01(value):
    return min(1.0, max(0.0, value))


def metric_value(log, key, source='evaluation'):
    entry = log.get(key)
    if not isinstance(entry, dict) or entry.get('agg_value') is None:
        raise ValueError(f'{source} requires {key}.agg_value in TOFU_EVAL.json')
    return float(entry['agg_value'])


def memorization_score(log):
    parts = [clip01(1.0 - metric_value(log, key))
             for key in ('extraction_strength', 'exact_memorization', 'forget_Q_A_Prob')]
    para = log['forget_Q_A_PARA_Prob']['value_by_index']
    pert = log['forget_Q_A_PERT_Prob']['value_by_index']
    ratios = []
    for idx, entry in para.items():
        if idx in pert:
            wrong = pert[idx]['prob']
            if isinstance(wrong, list):
                wrong = fmean(wrong)
            ratio = wrong / (entry['prob'] + EPS)
            ratios.append(min(ratio, 1.0 / (ratio + EPS)))
    if not ratios:
        raise ValueError('Mem requires matching paraphrase and perturbation example IDs')
    parts.append(clip01(fmean(ratios)))
    return hmean(parts)


def privacy_score(log, retain):
    similarities = []
    for key in MIA_KEYS:
        auc = metric_value(log, key)
        ref_auc = metric_value(retain, key, 'retain reference')
        if not (0 <= auc <= 1 and 0 <= ref_auc <= 1):
            raise ValueError(f'{key}: invalid AUC pair ({auc}, {ref_auc})')
        similarities.append(1.0 - abs(auc - ref_auc))
    return hmean(similarities)


def utility_score(log, retain):
    parts = []
    for key in UTILITY_KEYS:
        value = clip01(metric_value(log, key))
        reference = retain.get(key)
        if reference and reference.get('agg_value') is not None:
            value = min(value, clip01(float(reference['agg_value'])))
        parts.append(value)
    return hmean(parts)


def table_scores(log, retain):
    mem = memorization_score(log)
    priv = privacy_score(log, retain)
    util = utility_score(log, retain)
    return hmean((mem, priv, util)), mem, priv, util
