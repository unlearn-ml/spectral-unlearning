import re
import torch
import deepspeed
from trainer.unlearn.grad_diff import GradDiff

class RMU(GradDiff):

    def __init__(self, module_regex='model\\.layers\\.7', trainable_params_regex=['model\\.layers\\.(5|6|7)\\.mlp\\.down_proj\\.weight'], steering_coeff=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        self.trainable_params_regex = trainable_params_regex
        self.module_regex = module_regex
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.control_vec = None

    def create_optimizer(self):
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        super().create_optimizer()

    def _get_matching_module(self, model, module_regex):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        matched_modules = {name: module for name, module in model.named_modules() if re.fullmatch(module_regex, name)}
        if len(matched_modules) > 1:
            raise ValueError(f'More than one module matched with {module_regex}: {list(matched_modules.keys())}')
        elif not matched_modules:
            raise ValueError(f'No module matched with {module_regex}')
        return next(iter(matched_modules.values()))

    def _freeze_all_params(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        for name, param in model.named_parameters():
            if any((re.fullmatch(pattern, name) for pattern in trainable_params_regex)):
                param.requires_grad = requires_grad

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None
        hook_handle = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        hook_handle.remove()
        return (cache[0], outputs)

    def get_control_vector(self, dim):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = random_vector / torch.norm(random_vector) * self.steering_coeff
        return self.control_vec

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(activation1, activation2, reduction='none')
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        return (squared_diff_sum / num_tokens).mean()

    def compute_retain_loss(self, model, retain_inputs):
        retain_loss = 0.0
        if self.retain_loss_type == 'EMBED_DIFF':
            model_retain_activations, _ = self.forward_with_cache(model, retain_inputs, module=self.model_module, no_grad=False)
            ref_retain_activations, _ = self.forward_with_cache(self.ref_model, retain_inputs, module=self.ref_module, no_grad=True)
            mask = retain_inputs['labels'] != -100
            retain_loss = self.compute_activation_loss(model_retain_activations, ref_retain_activations.to(model_retain_activations.device), mask)
        else:
            retain_loss = super().compute_retain_loss(model, retain_inputs)
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs['forget']
        forget_inputs = {'input_ids': forget_inputs['input_ids'], 'attention_mask': forget_inputs['attention_mask'], 'labels': forget_inputs['labels']}
        model_forget_activations, forget_outputs = self.forward_with_cache(model, forget_inputs, self.model_module, no_grad=False)
        control_vec = forget_inputs.get('control_vec', self.get_control_vector(model_forget_activations.shape[-1]))
        control_vec = control_vec.to(dtype=model_forget_activations.dtype, device=model_forget_activations.device)
        control_vec = control_vec.expand_as(model_forget_activations)
        mask = forget_inputs['labels'] != -100
        forget_loss = self.compute_activation_loss(model_forget_activations, control_vec, mask)
        retain_inputs = inputs['retain']
        retain_inputs = {'input_ids': retain_inputs['input_ids'], 'attention_mask': retain_inputs['attention_mask'], 'labels': retain_inputs['labels']}
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        self._step_dynamics = {'forget_loss': float(forget_loss.detach().item()), 'retain_loss': float(retain_loss.detach().item()), 'steering_coeff': self.steering_coeff, 'gamma': self.gamma, 'alpha': self.alpha}
        return (loss, forget_outputs) if return_outputs else loss
