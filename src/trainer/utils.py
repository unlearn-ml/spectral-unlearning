import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])
    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])
    return (nn.functional.kl_div(current_probs, ref_probs, reduction='batchmean', log_target=True), outputs)

def compute_batch_nll(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs['labels']
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return (loss, outputs)

def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0, return_extras=False):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")
    win_log_ratio, lose_log_ratio = (0.0, 0.0)
    win_outputs, lose_outputs = (None, None)
    win_nll_mean, lose_nll_mean = (None, None)
    win_ref_nll_mean, lose_ref_nll_mean = (None, None)
    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)
        win_nll_mean = win_loss.detach().mean().item()
        win_ref_nll_mean = win_ref_loss.detach().mean().item()
    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
        lose_nll_mean = lose_loss.detach().mean().item()
        lose_ref_nll_mean = lose_ref_loss.detach().mean().item()
    margin = beta * (win_log_ratio - lose_log_ratio) if win_inputs is not None or lose_inputs is not None else 0.0
    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    if return_extras:
        extras = {'win_nll': win_nll_mean, 'lose_nll': lose_nll_mean, 'win_ref_nll': win_ref_nll_mean, 'lose_ref_nll': lose_ref_nll_mean, 'reward_margin': margin.detach().mean().item() if hasattr(margin, 'detach') else float(margin)}
        return (loss, (win_outputs, lose_outputs), extras)
    return (loss, (win_outputs, lose_outputs))

def compute_undial_loss(model, ref_model, inputs, beta):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs['labels']
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), soft_label.view(-1, soft_label.size(-1)))
    return (loss.mean(), outputs)

def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs['labels']
    labels = labels.to(outputs.logits.device)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    weight_ce = (-lm_loss).exp().detach() ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return (forget_loss, outputs)

def compute_satimp_loss(model, inputs, beta1, beta2, return_extras=False):
    outputs = model(**inputs)
    labels = inputs['labels']
    labels = labels.to(outputs.logits.device)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    valid_mask = shift_labels.view(-1) != -100
    weight_sat = (-lm_loss).exp().detach() ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    combined_weight = weight_sat * weight_imp
    forget_loss = -(combined_weight * lm_loss)[valid_mask].mean()
    if return_extras:
        extras = {'raw_lm_loss_mean': float(lm_loss[valid_mask].mean().item()), 'weight_sat_mean': float(weight_sat[valid_mask].mean().item()), 'weight_imp_mean': float(weight_imp[valid_mask].mean().item()), 'combined_weight_mean': float(combined_weight[valid_mask].mean().item()), 'combined_weight_max': float(combined_weight[valid_mask].max().item()), 'p_correct_mean': float((-lm_loss[valid_mask]).exp().mean().item())}
        return (forget_loss, outputs, extras)
    return (forget_loss, outputs)

class AdapterDisabledRef:

    def __init__(self, peft_model):
        self._peft = peft_model

    def __call__(self, **kwargs):
        with self._peft.disable_adapter():
            return self._peft(**kwargs)

    def forward(self, **kwargs):
        return self(**kwargs)

    def eval(self):
        return self

    def to(self, device):
        return self

    @property
    def device(self):
        return next(self._peft.parameters()).device
